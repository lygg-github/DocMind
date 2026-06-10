"""文本向量化服务 - 支持密集向量和稀疏向量（BM25），词表与 df 持久化 + 增量更新"""

# ========== 标准库导入 ==========
import json          # JSON序列化，用于持久化BM25状态到文件
import math          # 数学运算，用于计算IDF和对数
import os            # 操作系统接口，获取环境变量
import re            # 正则表达式，用于中文和英文分词
import threading     # 线程锁，保证并发安全
from collections import Counter  # 计数器，统计词频
from pathlib import Path        # 路径操作，管理状态文件路径

# ========== 第三方库导入 ==========
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace嵌入模型，用于生成稠密向量

# ========== 常量定义 ==========
# 默认的BM25状态持久化文件路径
# __file__：当前文件路径
# resolve()：解析为绝对路径
# parent.parent：向上两级到backend目录的父目录
# / "data" / "bm25_state.json"：拼接数据目录和状态文件名
_DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "bm25_state.json"

# ========== 全局注释 ==========
# 关键设计：所有 BM25 统计数据都存在内存，读写极快，仅在变更时持久化到磁盘
# 获得密集向量模型实例

# ========== 函数定义 ==========
def _create_dense_embedder() -> HuggingFaceEmbeddings:
    """
    创建稠密向量嵌入器（BGE-M3模型）
    
    Returns:
        HuggingFaceEmbeddings: 初始化好的嵌入模型实例
    
    Returns:
        HuggingFaceEmbeddings: 配置好的稠密向量嵌入模型
    """
    # 从环境变量获取模型名称，默认使用BAAI/bge-m3（多语言嵌入模型，支持中英文）
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    
    # 从环境变量获取设备类型，默认使用CPU（可选：cuda使用GPU）
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    
    # 创建HuggingFace嵌入模型实例
    # model_kwargs: 模型参数字典，device指定运行设备
    # encode_kwargs: 编码参数字典，normalize_embeddings=True将向量归一化到单位长度
    return HuggingFaceEmbeddings(
        model_name=model_name,                        # 模型名称或本地路径
        model_kwargs={"device": device},              # 设备配置：cpu或cuda
        encode_kwargs={"normalize_embeddings": True}, # 归一化：使向量长度为1，便于计算余弦相似度
    )


class EmbeddingService:
    """
    文本向量化服务类
    
    核心功能：
    1. 稠密向量生成：使用BGE-M3模型，生成1024维浮点向量（语义相似度检索）
    2. 稀疏向量生成：手写BM25算法，生成词袋稀疏向量（关键词精确匹配）
    3. 增量持久化：支持文档动态增删，BM25统计实时更新
    
    线程安全：通过threading.Lock保证多线程并发安全
    """
    
    def __init__(self, state_path: Path | str | None = None):
        """
        初始化向量化服务
        
        Args:
            state_path: BM25状态文件路径，为None时使用环境变量或默认路径
        """
        # 创建稠密向量嵌入器（BGE-M3）
        self._embedder = _create_dense_embedder()
        
        # 确定状态文件路径：
        # 1. 优先使用传入的state_path参数
        # 2. 其次使用环境变量BM25_STATE_PATH
        # 3. 最后使用默认路径 backend/data/bm25_state.json
        self._state_path = Path(state_path or os.getenv("BM25_STATE_PATH", _DEFAULT_STATE_PATH))
        
        # 创建线程锁，用于保护BM25统计数据的并发访问
        self._lock = threading.Lock()

        # ========== BM25 算法参数 ==========
        # BM25公式：score = IDF * (tf * (k1+1)) / (tf + k1 * (1-b + b * |D|/avgdl))
        
        # k1：词频饱和系数，控制词频对得分的影响程度
        # 值越大，越重视词频；值越小，词频影响越快饱和
        # 经验值：1.2~2.0，默认1.5
        self.k1 = 1.5
        
        # b：文档长度归一化系数，控制文档长度对得分的影响
        # 值越大，对长文档惩罚越重；值越小，文档长度影响越小
        # 经验值：0.5~1.0，默认0.75
        self.b = 0.75

        # ========== BM25 词表和统计变量 ==========
        # 词表：词字符串 → 词索引（稀疏向量的维度索引）
        # 例如：{"我": 0, "爱": 1, "北": 2, "京": 3}
        self._vocab: dict[str, int] = {}
        
        # 词表计数：下一个新词分配到的索引值（自增ID）
        # 每次添加新词后加1，保证词索引唯一
        self._vocab_counter = 0
        
        # 文档频率计数器：统计每个词出现在多少篇文档中
        # 例如：Counter({"我": 50, "爱": 30}) 表示"我"出现在50篇文档中
        self._doc_freq: Counter[str] = Counter()
        
        # 总文档数：已入库的文档（chunk）总数
        self._total_docs = 0
        
        # 所有文档的总词数：累加每篇文档的长度，用于计算平均长度
        self._sum_token_len = 0
        
        # 平均文档长度：总词数 / 文档数，用于BM25长度归一化
        self._avg_doc_len = 1.0

        # 从磁盘加载已持久化的BM25状态
        self._load_state()

    def _recompute_avg_len(self) -> None:
        """
        重新计算平均文档长度
        
        公式：avg_doc_len = sum_token_len / total_docs
        当文档数为0时，默认为1.0避免除零错误
        """
        # 如果文档数大于0，计算平均长度；否则默认为1.0
        self._avg_doc_len = (
            self._sum_token_len / self._total_docs if self._total_docs > 0 else 1.0
        )

    def _load_state(self) -> None:
        """
        从磁盘文件加载BM25持久化状态
        
        加载内容：
        - 词表（vocab）
        - 文档频率（doc_freq）
        - 总文档数（total_docs）
        - 总词数（sum_token_len）
        
        容错处理：
        - 文件不存在：跳过加载，使用初始空状态
        - JSON解析错误：跳过加载，使用初始空状态
        - 版本号不匹配：跳过加载，使用初始空状态
        """
        # 获取状态文件路径
        path = self._state_path
        
        # 如果文件不存在，直接返回（使用初始空状态）
        if not path.is_file():
            return
        
        # 尝试读取并解析JSON文件
        try:
            # read_text(encoding="utf-8")：读取文件内容为字符串，指定UTF-8编码
            # json.loads()：将JSON字符串解析为Python字典
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # JSON格式错误或IO错误：跳过加载
            return
        
        # 检查版本号，确保数据格式兼容
        # 当前版本为1，版本号不匹配时跳过加载
        if raw.get("version") != 1:
            return
        
        # ========== 加载词表 ==========
        # 遍历vocab字典，将键值对转换为字符串键和整数值
        # get("vocab", {})：如果不存在返回空字典
        # str(k)：确保键为字符串类型
        # int(v)：确保值为整数类型
        self._vocab = {str(k): int(v) for k, v in raw.get("vocab", {}).items()}
        
        # ========== 加载文档频率 ==========
        # 使用Counter包装，确保支持增加和减少操作
        self._doc_freq = Counter({str(k): int(v) for k, v in raw.get("doc_freq", {}).items()})
        
        # ========== 加载统计数字 ==========
        # 从JSON加载总文档数，默认为0
        self._total_docs = int(raw.get("total_docs", 0))
        
        # 从JSON加载总词数，默认为0
        self._sum_token_len = int(raw.get("sum_token_len", 0))
        
        # ========== 恢复词表计数器 ==========
        # 如果词表不为空，取词表最大值加1作为下一个词索引
        # 确保新词的索引不会与已有词冲突
        if self._vocab:
            # max(self._vocab.values())：获取最大索引值
            # +1：下一个词的起始索引
            self._vocab_counter = max(self._vocab.values()) + 1
        else:
            # 词表为空时，计数器归零
            self._vocab_counter = 0
        
        # 重新计算平均文档长度
        self._recompute_avg_len()

    # ========== 持久化方法 ==========
    # 这两个方法用于将BM25状态保存到磁盘
    
    def _persist_unlocked(self) -> None:
        """
        将BM25状态持久化到磁盘（无锁版本）
        
        持久化策略：
        1. 先写入临时文件（.tmp后缀）
        2. 写入成功后，原子替换原文件
        3. 避免写入过程中程序崩溃导致文件损坏
        
        持久化内容：
        - version：数据版本号（用于格式兼容）
        - total_docs：总文档数
        - sum_token_len：总词数
        - vocab：词表字典
        - doc_freq：文档频率字典
        """
        # 确保父目录存在（parents=True创建多级目录，exist_ok=True目录已存在不报错）
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建持久化数据字典
        payload = {
            "version": 1,                    # 版本号，用于未来格式升级兼容
            "total_docs": self._total_docs,    # 总文档数
            "sum_token_len": self._sum_token_len,  # 总词数
            "vocab": self._vocab,              # 词表：词→索引映射
            "doc_freq": dict(self._doc_freq),  # 文档频率：词→出现文档数
        }
        
        # 构建临时文件路径：在原路径后加.tmp后缀
        tmp = self._state_path.with_suffix(".json.tmp")
        
        # 写入临时文件
        # ensure_ascii=False：允许中文字符直接写入（不转义为\u开头的ASCII）
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        
        # 原子替换：临时文件写入成功后，用它替换原文件
        # 这是原子操作，避免文件损坏
        tmp.replace(self._state_path)

    def _persist(self) -> None:
        """
        将BM25状态持久化到磁盘（加锁版本）
        
        线程安全包装：
        - 获取锁后调用无锁版本
        - 确保多线程并发调用时数据一致
        """
        # 获取线程锁
        with self._lock:
            # 调用无锁版本的持久化方法
            self._persist_unlocked()

    # ========== 对称设计 ==========
    # 添加 / 删除文档完全对应，支持语料动态维护
    # 核心逻辑：
    # 对文本分词
    # 更新总文档数、文档总长度
    # 更新文档频率（df）：一个词在多少篇文档中出现
    # 重新计算平均文档长度
    # 持久化到磁盘
    # 设计亮点：删除文档时不回收词表索引，避免向量维度冲突（适配 Milvus 等向量库）
    
    def increment_add_documents(self, texts: list[str]) -> None:
        """
        增量添加文档时，更新BM25统计信息
        
        调用时机：文档入库时调用（写入Milvus之前或之后）
        
        Args:
            texts: 文本列表，每个文本视为一篇独立的BM25文档
        
        更新内容：
        1. 总文档数 +1
        2. 总词数累加
        3. 词表扩展（新词分配索引）
        4. 文档频率 +1（每个词在多少文档中出现）
        5. 重新计算平均长度
        6. 持久化到磁盘
        """
        # 空列表处理：直接返回，避免无意义操作
        if not texts:
            return
        
        # 获取线程锁，保证并发安全
        with self._lock:
            # 遍历每个文本
            for text in texts:
                # 1. 对文本进行分词
                tokens = self.tokenize(text)
                
                # 2. 记录文档长度
                doc_len = len(tokens)
                
                # 3. 累加总词数
                self._sum_token_len += doc_len
                
                # 4. 文档数 +1
                self._total_docs += 1
                
                # 5. 更新词表和文档频率
                # set(tokens)：去重，每个词只计数一次（文档频率是"出现过的文档数"，不是"出现次数"）
                for token in set(tokens):
                    # 如果词不在词表中，分配新索引
                    if token not in self._vocab:
                        self._vocab[token] = self._vocab_counter
                        self._vocab_counter += 1
                    
                    # 该词的文档频率 +1
                    self._doc_freq[token] += 1
            
            # 6. 重新计算平均文档长度
            self._recompute_avg_len()
            
            # 7. 持久化更新后的状态到磁盘
            self._persist_unlocked()

    def increment_remove_documents(self, texts: list[str]) -> None:
        """
        增量删除文档时，对称扣减BM25统计信息
        
        调用时机：文档删除时调用（从Milvus删除之前或之后）
        
        Args:
            texts: 文本列表，每个文本视为一篇待移除的BM25文档
        
        扣减内容：
        1. 总文档数 -1（使用max防止负数）
        2. 总词数扣减（使用max防止负数）
        3. 文档频率 -1（频率为0时清除该词）
        4. 重新计算平均长度
        5. 持久化到磁盘
        
        设计亮点：
        - 词表索引不回收，避免与Milvus中已存在的旧稀疏向量维度冲突
        - 这是"只增不减"设计，保证数据一致性
        """
        # 空列表处理
        if not texts:
            return
        
        with self._lock:
            for text in texts:
                # 1. 对文本进行分词
                tokens = self.tokenize(text)
                
                # 2. 记录文档长度
                doc_len = len(tokens)
                
                # 3. 扣减总词数（max防止负数）
                self._sum_token_len = max(0, self._sum_token_len - doc_len)
                
                # 4. 扣减文档数（max防止负数）
                self._total_docs = max(0, self._total_docs - 1)
                
                # 5. 扣减文档频率
                for token in set(tokens):
                    # 如果该词不在统计中，跳过
                    if token not in self._doc_freq:
                        continue
                    
                    # 文档频率 -1
                    self._doc_freq[token] -= 1
                    
                    # 如果频率降为0或负数，清除该词记录
                    # 注意：词表（_vocab）中的索引不回收！
                    if self._doc_freq[token] <= 0:
                        del self._doc_freq[token]
            
            # 重新计算平均文档长度
            self._recompute_avg_len()
            
            # 持久化更新后的状态
            self._persist_unlocked()

    # ========== 稠密向量生成方法 ==========
    
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        使用BGE-M3模型生成稠密向量
        
        Args:
            texts: 文本列表
        
        Returns:
            二维浮点数列表：每个文本对应一个向量（维度通常为1024）
        
        特点：
        - 语义向量：捕捉文本的语义信息
        - 归一化：向量长度为1，便于计算余弦相似度
        - 适用场景：语义相似度检索、聚类、分类等
        """
        # 空列表处理
        if not texts:
            return []
        
        try:
            # 调用HuggingFace嵌入模型的文档嵌入方法
            # embed_documents：批量嵌入多个文档
            return self._embedder.embed_documents(texts)
        except Exception as e:
            # 捕获异常并重新抛出，添加中文错误信息
            raise Exception(f"本地嵌入模型调用失败: {str(e)}") from e

    # ========== 中英文混合分词策略 ==========
    # 中文：单字分词（适合中文 BM25，效果优于词级分词）
    # 英文：单词分词（连续字母作为一个词）
    # 忽略标点、数字、符号
    # 统一转小写，消除大小写差异
    
    def tokenize(self, text: str) -> list[str]:
        """
        中英文混合分词
        
        分词策略：
        - 中文：单字分词（如"企业知识管理" → ["企", "业", "知", "识", "管", "理"]）
        - 英文：单词分词（如"RAG system" → ["rag", "system"]）
        - 其他：忽略（标点、数字、符号等）
        
        Args:
            text: 待分词文本
        
        Returns:
            词列表
        
        设计亮点：
        - 中文单字分词：适合中文BM25，因为中文词边界不明确
        - 英文单词分词：标准做法，便于计算词频
        - 统一小写：消除大小写差异，"RAG"和"rag"视为同一词
        """
        # 统一转换为小写，消除大小写差异
        text = text.lower()
        
        # 初始化结果列表
        tokens = []
        
        # 编译正则表达式（提高匹配效率）
        # 中文正则：匹配Unicode范围\u4e00-\u9fff内的字符（基本汉字）
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        
        # 英文正则：匹配连续一个或多个字母
        english_pattern = re.compile(r"[a-zA-Z]+")
        
        # 初始化字符索引
        i = 0
        
        # 遍历文本的每个字符
        while i < len(text):
            # 获取当前字符
            char = text[i]
            
            # 判断是否为中文字符
            if chinese_pattern.match(char):
                # 中文字符：直接添加为单独的词
                tokens.append(char)
                i += 1  # 移动到下一个字符
            
            # 判断是否为一个或多个连续英文字母
            elif english_pattern.match(char):
                # 从当前位置匹配连续英文字母
                match = english_pattern.match(text[i:])
                if match:
                    # 添加整个英文单词
                    tokens.append(match.group())
                    # 移动索引，跳过整个单词
                    i += len(match.group())
            
            # 其他字符（标点、数字、符号等）：直接跳过
            else:
                i += 1
        
        return tokens

    # ========== 稀疏向量（BM25）生成 ==========
    # 稀疏向量（BM25）生成（核心算法）
    # 步骤：
    # 分词 → 统计词频（tf）
    # 计算逆文档频率（IDF）：词越稀有，权重越高
    # 计算词的 BM25 得分
    # 生成稀疏向量：{词索引：得分}
    # 自动将新词加入词表，标记是否需要持久化
    # 对外接口：
    # get_sparse_embedding()：单文本稀疏向量
    # get_sparse_embeddings()：批量文本稀疏向量
    # get_all_embeddings()：一次性返回密集 + 稀疏向量（最高效）
    
    def _sparse_vector_for_text_unlocked(self, text: str) -> tuple[dict, bool]:
        """
        为单个文本生成BM25稀疏向量（无锁版本）
        
        算法流程：
        1. 分词：调用tokenize方法
        2. 统计词频（tf）：每个词在该文档中出现多少次
        3. 计算IDF：逆文档频率，词越稀有分数越高
        4. 计算BM25得分：综合词频和文档长度
        5. 生成稀疏向量：{词索引：得分}
        6. 动态词表：新词自动加入词表
        
        Args:
            text: 待处理的文本
        
        Returns:
            tuple[dict, bool]: 
            - dict: 稀疏向量，格式为 {词索引: BM25得分}
            - bool: 词表是否发生变化（用于判断是否需要持久化）
        
        BM25公式详解：
        score = IDF * (tf * (k1+1)) / (tf + k1 * (1-b + b * |D|/avgdl))
        
        公式各部分含义：
        - IDF：逆文档频率，log((N-n+0.5)/(n+0.5)+1)
        - tf：词在该文档中出现的频率
        - |D|：文档长度（词数）
        - avgdl：平均文档长度
        - k1：词频饱和参数（默认1.5）
        - b：长度归一化参数（默认0.75）
        """
        # 1. 分词
        tokens = self.tokenize(text)
        
        # 2. 记录文档长度（用于长度归一化）
        doc_len = len(tokens)
        
        # 3. 统计词频（tf）
        # Counter会统计每个词出现的次数
        # 例如：Counter({"我": 2, "爱": 1, "北": 1, "京": 1})
        tf = Counter(tokens)
        
        # 4. 初始化稀疏向量
        # 格式：{词索引: BM25得分}
        # 稀疏向量只存储非零元素，节省存储空间
        sparse_vector: dict[int, float] = {}
        
        # 5. 标记词表是否发生变化（新词加入时需要持久化）
        vocab_changed = False
        
        # 获取总文档数（用于IDF计算），避免除零
        n = max(self._total_docs, 0)
        
        # 获取平均文档长度（用于长度归一化），避免除零
        avg = max(self._avg_doc_len, 1.0)

        # 6. 遍历每个词，计算BM25得分
        for token, freq in tf.items():
            # 如果词不在词表中，分配新索引（新词）
            if token not in self._vocab:
                self._vocab[token] = self._vocab_counter
                self._vocab_counter += 1
                vocab_changed = True  # 标记词表已变化

            # 获取该词的索引
            idx = self._vocab[token]
            
            # 获取该词的文档频率（df）
            # df：该词出现在多少篇文档中
            df = self._doc_freq.get(token, 0)
            
            # ========== 计算IDF（逆文档频率） ==========
            # IDF含义：一个词越稀有，在检索时权重越高
            # 例如："的"几乎每篇文档都出现，IDF很低
            #       "量子"只有少数文档出现，IDF很高
            if df == 0:
                # 特殊情况：词从未出现过，使用平滑公式
                # log((N+1)/1) = log(N+1)
                idf = math.log((n + 1) / 1)
            else:
                # 标准IDF公式（带平滑）
                # log((N-df+0.5)/(df+0.5)+1)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            # ========== 计算BM25得分 ==========
            # 分子：词频部分，freq * (k1 + 1)
            numerator = freq * (self.k1 + 1)
            
            # 分母：词频 + 长度归一化
            # k1 * (1 - b + b * doc_len / avg)
            # - 当文档长度等于平均长度时：1 - b + b * 1 = 1
            # - 当文档长度大于平均长度时：1 - b + b * 1.5 > 1（惩罚长文档）
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / avg)
            
            # 最终得分 = IDF * 分子 / 分母
            score = idf * numerator / denominator
            
            # 只保存得分大于0的词（避免无意义数据）
            if score > 0:
                sparse_vector[idx] = float(score)

        # 返回稀疏向量和词表变化标记
        return sparse_vector, vocab_changed

    def get_sparse_embedding(self, text: str) -> dict:
        """
        获取单个文本的稀疏向量（BM25）
        
        Args:
            text: 待处理的文本
        
        Returns:
            dict: 稀疏向量，格式为 {词索引: BM25得分}
        
        特点：
        - 线程安全：使用锁保护
        - 自动持久化：当词表变化时自动保存状态
        """
        with self._lock:
            # 调用无锁版本生成稀疏向量
            sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
            
            # 如果词表发生变化（新增了词），持久化状态
            if vocab_changed:
                self._persist_unlocked()
        
        return sparse_vector

    def get_sparse_embeddings(self, texts: list[str]) -> list[dict]:
        """
        批量获取多个文本的稀疏向量
        
        Args:
            texts: 文本列表
        
        Returns:
            list[dict]: 稀疏向量列表，每个元素对应一个文本的稀疏向量
        
        特点：
        - 高效：一次处理多个文本，减少锁竞争
        - 批量持久化：只要有任何一个文本引入了新词，就持久化一次
        """
        # 空列表处理
        if not texts:
            return []
        
        with self._lock:
            # 初始化结果列表
            out: list[dict] = []
            
            # 标记是否有任何文本引入了新词
            any_new_vocab = False
            
            # 遍历每个文本
            for text in texts:
                # 生成稀疏向量
                sparse_vector, vocab_changed = self._sparse_vector_for_text_unlocked(text)
                
                # 添加到结果列表
                out.append(sparse_vector)
                
                # 更新新词标记（任意一个为True就为True）
                any_new_vocab = any_new_vocab or vocab_changed
            
            # 如果有新增词汇，持久化状态
            if any_new_vocab:
                self._persist_unlocked()
        
        return out

    def get_all_embeddings(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
        """
        同时获取稠密向量和稀疏向量（最高效的批量接口）
        
        Args:
            texts: 文本列表
        
        Returns:
            tuple:
            - list[list[float]]: 稠密向量列表（BGE-M3生成）
            - list[dict]: 稀疏向量列表（BM25生成）
        
        特点：
        - 一次性返回两种向量，避免重复分词
        - 推荐用于混合检索场景
        """
        # 调用稠密向量生成方法
        dense_embeddings = self.get_embeddings(texts)
        
        # 调用稀疏向量批量生成方法
        sparse_embeddings = self.get_sparse_embeddings(texts)
        
        # 返回两个向量列表
        return dense_embeddings, sparse_embeddings


# ========== 全局单例 ==========
# 全进程唯一实例：写入与检索共用同一份 BM25 持久化状态
# 作用：
# 1. 保证所有模块使用同一份BM25统计
# 2. 避免重复加载模型和状态
# 3. 提供全局访问点

# 实例化全局唯一的EmbeddingService对象
# 特点：
# - 延迟初始化：首次导入时创建
# - 线程安全：内部使用锁保护共享状态
embedding_service = EmbeddingService()
