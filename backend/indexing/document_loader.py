"""文档加载和分片服务"""

# ========== 标准库导入 ==========
import os  # 操作系统接口，用于文件路径操作和目录遍历

# ========== 类型提示导入 ==========
from typing import Dict, List  # Dict：字典类型提示，List：列表类型提示

# ========== LangChain 文档加载器导入 ==========
# Docx2txtLoader：Word文档加载器（.docx格式）
# PyPDFLoader：PDF文档加载器
# UnstructuredExcelLoader：Excel文档加载器（.xlsx/.xls格式）
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader

# ========== LangChain 文本分割器导入 ==========
# RecursiveCharacterTextSplitter：递归字符文本分割器
# 按照分隔符优先级递归切分文本，保持语义完整性
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """
    文档加载和分片服务类
    
    核心功能：
    1. 多格式文档加载：支持PDF、Word、Excel、HTML
    2. 三级递归分块：L1（大块）→ L2（中块）→ L3（小块）
    3. 层级关系维护：父子块ID关联，支持Auto-merging
    
    三级分块设计：
    - L1：~2400字符，大段落级别，作为根节点
    - L2：~1600字符，小节级别，L1的子节点
    - L3：~800字符，句子级别，L2的子节点，叶子节点
    
    存储策略：
    - L1/L2：存入PostgreSQL（父块存储）
    - L3：存入Milvus（向量检索）
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        初始化文档加载器，配置三级分块参数
        
        Args:
            chunk_size: 基础块大小（L3级别），默认800字符
            chunk_overlap: 块重叠大小，默认100字符（保持上下文连贯）
        
        三级分块参数计算：
        - L1：最大块，chunk_size × 3
        - L2：中块，chunk_size × 2
        - L3：小块，chunk_size × 1
        """
        # ========== 计算三级分块参数 ==========
        
        # L1级别（最大块）：至少2000字符，或chunk_size的3倍
        # 用途：作为根节点，存储完整段落
        level_1_size = max(2000, chunk_size * 3)  # 默认2400字符
        
        # L1级别重叠：至少400字符，或chunk_overlap的3倍
        # 重叠区域保证上下文连贯，避免语义被切断
        level_1_overlap = max(400, chunk_overlap * 3)  # 默认400字符
        
        # L2级别（中块）：至少1000字符，或chunk_size的2倍
        # 用途：作为L1的子节点，存储小节内容
        level_2_size = max(1000, chunk_size * 2)  # 默认1600字符
        
        # L2级别重叠：至少200字符，或chunk_overlap的2倍
        level_2_overlap = max(200, chunk_overlap * 2)  # 默认200字符
        
        # L3级别（小块）：至少600字符，或chunk_size
        # 用途：作为L2的子节点，叶子节点，入Milvus向量检索
        level_3_size = max(600, chunk_size)  # 默认800字符
        
        # L3级别重叠：至少100字符，或chunk_overlap
        level_3_overlap = max(100, chunk_overlap)  # 默认100字符

        # ========== 创建三级文本分割器 ==========
        
        # L1级别分割器（最大块）
        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,           # 块大小：~2400字符
            chunk_overlap=level_1_overlap,     # 重叠大小：~400字符
            add_start_index=True,              # 添加起始索引到元数据（用于定位原文位置）
            # 分隔符优先级（从高到低）：
            # "\n\n"：段落分隔（最高优先级，保持段落完整）
            # "。"：中文句号
            # "！"：中文感叹号
            # "？"：中文问号
            # "\n"：换行符
            # "，"：中文逗号
            # "、"：中文顿号
            # " "：空格
            # ""：空字符串（最后兜底，强制分割）
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],
        )
        
        # L2级别分割器（中块）
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,           # 块大小：~1600字符
            chunk_overlap=level_2_overlap,     # 重叠大小：~200字符
            add_start_index=True,              # 添加起始索引
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],  # 同L1
        )
        
        # L3级别分割器（小块，叶子节点）
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,           # 块大小：~800字符
            chunk_overlap=level_3_overlap,     # 重叠大小：~100字符
            add_start_index=True,              # 添加起始索引
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],  # 同L1
        )

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        """
        构建分块唯一标识符
        
        ID格式：{filename}::p{page_number}::l{level}::{index}
        例如：report.pdf::p1::l3::5
        
        Args:
            filename: 文件名
            page_number: 页码
            level: 分块层级（1/2/3）
            index: 该层级内的序号
        
        Returns:
            str: 唯一的分块ID字符串
        
        设计目的：
        - 全局唯一：文件名+页码+层级+序号组合保证唯一
        - 可解析：通过::分隔符可反解析各字段
        - 可读性：ID本身包含位置信息，便于调试
        """
        # 使用::作为分隔符，拼接各字段
        # f-string格式化：{filename}::p{page_number}::l{level}::{index}
        return f"{filename}::p{page_number}::l{level}::{index}"

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        """
        将单页文本进行三级递归分块
        
        核心算法：
        1. L1切分：将整页文本切分为多个大块
        2. L2切分：对每个L1块进行二次切分
        3. L3切分：对每个L2块进行三次切分
        
        层级关系：
        - L1：根节点，parent_chunk_id=""，root_chunk_id=自身ID
        - L2：L1的子节点，parent_chunk_id=L1的ID，root_chunk_id=L1的ID
        - L3：L2的子节点，parent_chunk_id=L2的ID，root_chunk_id=L1的ID
        
        Args:
            text: 待分块的文本内容（单页）
            base_doc: 基础元数据字典（filename、page_number等）
            page_global_chunk_idx: 当前页的全局分块起始索引
        
        Returns:
            List[Dict]: 所有层级的分块列表（包含L1、L2、L3）
        """
        # 空文本处理：直接返回空列表
        if not text:
            return []

        # ========== 初始化变量 ==========
        
        # 结果列表：存储所有层级的分块
        root_chunks: List[Dict] = []
        
        # 获取页码（从base_doc中提取，默认为0）
        page_number = int(base_doc.get("page_number", 0))
        
        # 获取文件名（从base_doc中提取）
        filename = base_doc["filename"]

        # ========== L1级别切分 ==========
        
        # 使用L1分割器对整页文本进行切分
        # create_documents：返回Document对象列表，每个包含page_content和metadata
        # 参数：[text]文本列表，[base_doc]元数据列表
        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        
        # L1级别计数器：用于生成L1的chunk_id
        level_1_counter = 0
        
        # L2级别计数器：用于生成L2的chunk_id
        level_2_counter = 0
        
        # L3级别计数器：用于生成L3的chunk_id
        level_3_counter = 0

        # ========== 遍历L1分块 ==========
        for level_1_doc in level_1_docs:
            # 获取L1块的文本内容（去除首尾空白）
            level_1_text = (level_1_doc.page_content or "").strip()
            
            # 空文本跳过
            if not level_1_text:
                continue
            
            # 生成L1块的唯一ID
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            
            # L1计数器+1
            level_1_counter += 1

            # ========== 构建L1分块字典 ==========
            level_1_chunk = {
                **base_doc,                        # 展开基础元数据（filename、page_number等）
                "text": level_1_text,              # 分块文本内容
                "chunk_id": level_1_id,            # 分块唯一ID
                "parent_chunk_id": "",             # L1是根节点，无父节点
                "root_chunk_id": level_1_id,       # 根节点ID指向自身
                "chunk_level": 1,                  # 层级：1
                "chunk_idx": page_global_chunk_idx,  # 全局索引
            }
            
            # 全局索引+1
            page_global_chunk_idx += 1
            
            # 将L1块添加到结果列表
            root_chunks.append(level_1_chunk)

            # ========== L2级别切分（对当前L1块进行切分） ==========
            
            # 使用L2分割器对L1文本进行切分
            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            
            # 遍历L2分块
            for level_2_doc in level_2_docs:
                # 获取L2块的文本内容
                level_2_text = (level_2_doc.page_content or "").strip()
                
                # 空文本跳过
                if not level_2_text:
                    continue
                
                # 生成L2块的唯一ID
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                
                # L2计数器+1
                level_2_counter += 1

                # ========== 构建L2分块字典 ==========
                level_2_chunk = {
                    **base_doc,                        # 展开基础元数据
                    "text": level_2_text,              # 分块文本内容
                    "chunk_id": level_2_id,            # 分块唯一ID
                    "parent_chunk_id": level_1_id,     # 父节点ID指向L1
                    "root_chunk_id": level_1_id,       # 根节点ID指向L1
                    "chunk_level": 2,                  # 层级：2
                    "chunk_idx": page_global_chunk_idx,  # 全局索引
                }
                
                # 全局索引+1
                page_global_chunk_idx += 1
                
                # 将L2块添加到结果列表
                root_chunks.append(level_2_chunk)

                # ========== L3级别切分（对当前L2块进行切分） ==========
                
                # 使用L3分割器对L2文本进行切分
                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                
                # 遍历L3分块
                for level_3_doc in level_3_docs:
                    # 获取L3块的文本内容
                    level_3_text = (level_3_doc.page_content or "").strip()
                    
                    # 空文本跳过
                    if not level_3_text:
                        continue
                    
                    # 生成L3块的唯一ID
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    
                    # L3计数器+1
                    level_3_counter += 1
                    
                    # ========== 构建L3分块字典并添加到结果 ==========
                    # 硬截断保护：Milvus text 字段 max_length=2000，超长会被拒绝
                    level_3_text_safe = level_3_text[:2000]
                    root_chunks.append({
                        **base_doc,                        # 展开基础元数据
                        "text": level_3_text_safe,         # 分块文本内容（安全截断至2000字符）
                        "chunk_id": level_3_id,            # 分块唯一ID
                        "parent_chunk_id": level_2_id,     # 父节点ID指向L2
                        "root_chunk_id": level_1_id,       # 根节点ID指向L1
                        "chunk_level": 3,                  # 层级：3（叶子节点）
                        "chunk_idx": page_global_chunk_idx,  # 全局索引
                    })
                    
                    # 全局索引+1
                    page_global_chunk_idx += 1

        # 返回所有层级的分块列表
        return root_chunks

    def _load_from_langchain_docs(
        self,
        raw_docs: list,
        file_path: str,
        filename: str,
        doc_type: str,
    ) -> list[dict]:
        """
        将LangChain文档对象转换为三级分块字典列表
        
        处理流程：
        1. 遍历每个原始文档（通常每页一个Document）
        2. 提取页码和元数据
        3. 调用_split_page_to_three_levels进行三级分块
        4. 合并所有分块结果
        
        Args:
            raw_docs: LangChain Document对象列表
            file_path: 文件完整路径
            filename: 文件名
            doc_type: 文档类型（PDF/Word/Excel/HTML）
        
        Returns:
            list[dict]: 所有页面的三级分块列表
        """
        # 初始化结果列表
        documents: list[dict] = []
        
        # 全局分块索引（跨页面累计）
        page_global_chunk_idx = 0
        
        # 遍历每个原始文档（每个Document通常对应一页）
        for doc in raw_docs:
            # 获取文档元数据（使用getattr安全获取，默认为空字典）
            meta = getattr(doc, "metadata", None) or {}
            
            # 从元数据中获取页码（PDF通常有page字段）
            page_num = meta.get("page", 0)
            
            # None值处理：转换为0
            if page_num is None:
                page_num = 0
            
            # 类型转换：确保页码为整数
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                # 转换失败时默认为0
                page_num = 0
            
            # ========== 构建基础元数据字典 ==========
            base_doc = {
                "filename": filename,       # 文件名
                "file_path": file_path,     # 文件完整路径
                "file_type": doc_type,      # 文档类型（PDF/Word/Excel/HTML）
                "page_number": page_num,    # 页码
            }
            
            # ========== 对当前页面进行三级分块 ==========
            page_chunks = self._split_page_to_three_levels(
                text=(doc.page_content or "").strip(),  # 文档内容（去除首尾空白）
                base_doc=base_doc,                       # 基础元数据
                page_global_chunk_idx=page_global_chunk_idx,  # 全局起始索引
            )
            
            # 更新全局索引（加上当前页面的分块数量）
            page_global_chunk_idx += len(page_chunks)
            
            # 将当前页面的分块添加到结果列表
            documents.extend(page_chunks)
        
        # 返回所有页面的分块结果
        return documents

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """
        加载单个文档并进行三级分块
        
        支持格式：
        - PDF：使用PyPDFLoader
        - Word：使用Docx2txtLoader
        - Excel：使用UnstructuredExcelLoader
        - HTML：使用自定义html_processor
        
        Args:
            file_path: 文件完整路径
            filename: 文件名（用于判断文件类型）
        
        Returns:
            list[dict]: 三级分块字典列表
        
        Raises:
            ValueError: 不支持的文件类型
            Exception: 文档处理失败
        """
        # 将文件名转换为小写（便于后缀匹配）
        file_lower = filename.lower()

        # ========== 根据文件后缀选择加载器 ==========
        
        # PDF文件处理
        if file_lower.endswith(".pdf"):
            doc_type = "PDF"  # 文档类型标识
            loader = PyPDFLoader(file_path)  # 创建PDF加载器
        
        # Word文件处理（.docx或.doc）
        elif file_lower.endswith((".docx", ".doc")):
            doc_type = "Word"  # 文档类型标识
            loader = Docx2txtLoader(file_path)  # 创建Word加载器
        
        # Excel文件处理（.xlsx或.xls）
        elif file_lower.endswith((".xlsx", ".xls")):
            doc_type = "Excel"  # 文档类型标识
            loader = UnstructuredExcelLoader(file_path)  # 创建Excel加载器
        
        # HTML文件处理（.html或.htm）
        elif file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"  # 文档类型标识
            
            # 延迟导入HTML处理器（避免循环依赖）
            from backend.indexing.html_processor import load_html_for_document_loader

            # 使用自定义HTML处理器加载文档
            raw_docs = load_html_for_document_loader(file_path, filename)
            
            # 直接返回处理结果（HTML处理器已返回Document列表）
            return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type)
        
        # 不支持的文件类型
        else:
            raise ValueError(f"不支持的文件类型: {filename}")

        # ========== 使用LangChain加载器加载文档 ==========
        try:
            # 调用加载器的load方法，返回Document对象列表
            raw_docs = loader.load()
            
            # 将Document对象转换为三级分块字典
            return self._load_from_langchain_docs(raw_docs, file_path, filename, doc_type)
        
        except Exception as e:
            # 捕获异常并重新抛出，添加中文错误信息
            raise Exception(f"处理文档失败: {str(e)}") from e

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """
        从文件夹批量加载所有支持的文档
        
        扫描文件夹中所有PDF、Word、Excel、HTML文件，
        逐个加载并进行三级分块，合并所有结果。
        
        Args:
            folder_path: 文件夹路径
        
        Returns:
            list[dict]: 所有文档的三级分块列表
        
        注意：
        - 跳过不支持的文件类型
        - 跳过加载失败的文件（静默失败）
        """
        # 初始化结果列表
        all_documents = []

        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 将文件名转换为小写（便于后缀匹配）
            file_lower = filename.lower()
            
            # ========== 检查文件类型是否支持 ==========
            # 使用逻辑或组合多个条件
            if not (
                file_lower.endswith(".pdf")              # PDF文件
                or file_lower.endswith((".docx", ".doc"))  # Word文件
                or file_lower.endswith((".xlsx", ".xls"))  # Excel文件
                or file_lower.endswith((".html", ".htm"))  # HTML文件
            ):
                continue  # 跳过不支持的文件类型

            # 拼接完整文件路径
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 加载单个文档并进行三级分块
                documents = self.load_document(file_path, filename)
                
                # 将结果添加到总列表
                all_documents.extend(documents)
            
            except Exception:
                # 静默失败：跳过加载失败的文件
                continue

        # 返回所有文档的分块结果
        return all_documents
