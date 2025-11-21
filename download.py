from argparse import ArgumentParser
import time
import re
import http.client
from Bio import Entrez
import json

def setup_http_protocol():
    """解决HTTP协议版本问题"""
    http.client.HTTPConnection._http_vsn = 10
    http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

def validate_date(date_str):
    """Validate date format YYYY/MM/DD"""
    if not re.match(r'\d{4}/\d{2}/\d{2}', date_str):
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY/MM/DD")
    return date_str

def search(query, max_num_articles, email):
    """检索PubMed文章ID"""
    Entrez.email = email
    try:
        handle = Entrez.esearch(
            db='pubmed',
            sort='relevance',
            retmax=max_num_articles,
            retmode='xml',
            term=query,
            usehistory="y"
        )
        results = Entrez.read(handle)
        handle.close()
        return results
    except Exception as e:
        print(f"搜索错误: {e}")
        return None

def fetch_details(id_list, retry_count=3):
    """获取文章详情，包含重试机制"""
    for attempt in range(retry_count):
        try:
            ids = ','.join(id_list)
            handle = Entrez.efetch(
                db='pubmed',
                retmode='xml',
                id=ids
            )
            results = Entrez.read(handle)
            handle.close()
            return results
        except Exception as e:
            if attempt == retry_count - 1:
                print(f"获取详情失败: {e}")
                return None
            time.sleep(2)

def extract_article_info(paper):
    """从文章数据中提取结构化信息"""
    try:
        article = paper['MedlineCitation']['Article']

        # 提取标题
        title = article.get('ArticleTitle', 'No title available')

        # 提取摘要
        abstract = article.get('Abstract', {})
        abstract_text = ''
        if abstract and 'AbstractText' in abstract:
            if isinstance(abstract['AbstractText'], list):
                abstract_text = ' '.join([str(text) for text in abstract['AbstractText']])
            else:
                abstract_text = str(abstract['AbstractText'])

        # 提取日期信息
        pub_date = extract_publication_date(article)

        # 提取作者信息
        authors = extract_authors(article)

        # 提取期刊信息
        journal_info = extract_journal_info(article)

        return {
            "pmid": paper['MedlineCitation']['PMID'],
            "title": title,
            "abstract": abstract_text,
            "pub_date": pub_date,
            "authors": authors,
            "journal": journal_info
        }
    except Exception as e:
        print(f"提取文章信息错误: {e}")
        return None

def extract_publication_date(article):
    """提取出版日期"""
    date_fields = ['ArticleDate', 'JournalIssue', 'PubDate']
    for field in date_fields:
        if field in article:
            date_info = article[field]
            if isinstance(date_info, list) and len(date_info) > 0:
                date_info = date_info[0]
            if 'Year' in date_info:
                return {
                    "year": date_info.get('Year', ''),
                    "month": date_info.get('Month', ''),
                    "day": date_info.get('Day', '')
                }
    return {"year": "", "month": "", "day": ""}

def extract_authors(article):
    """提取作者信息"""
    authors = []
    if 'AuthorList' in article:
        for author in article['AuthorList']:
            author_name = f"{author.get('LastName', '')} {author.get('ForeName', '')}".strip()
            if author_name:
                authors.append(author_name)
    return authors

def extract_journal_info(article):
    """提取期刊信息"""
    journal = article.get('Journal', {})
    return {
        "title": journal.get('Title', ''),
        "volume": journal.get('JournalIssue', {}).get('Volume', ''),
        "issue": journal.get('JournalIssue', {}).get('Issue', '')
    }

def get_pubmed_data(
        output_json_file: str,
        start_date: str = "2023/12/1",
        end_date: str = "2023/12/31",
        email: str = '2843579506@qq.com',
        max_num_articles: int = 1000
):
    """
    下载指定日期范围内的PubMed文章摘要
    """
    setup_http_protocol()

    try:
        validate_date(start_date)
        validate_date(end_date)
    except ValueError as e:
        print(f"日期格式错误: {e}")
        return

    if email == 'your_email@example.com':
        print("警告: 请将email参数替换为您的实际邮箱地址")
        return

    Entrez.email = email

    query = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'

    print(f"开始搜索PubMed文章: {query}")
    start_time = time.time()

    results = search(query, max_num_articles, email)
    if not results or 'IdList' not in results:
        print("未找到文章")
        return

    id_list = results['IdList']
    print(f"找到 {len(id_list)} 篇文章")

    batch_size = 200
    all_articles = []

    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(id_list)-1)//batch_size + 1}")

        papers = fetch_details(batch_ids)
        if papers and 'PubmedArticle' in papers:
            for paper in papers['PubmedArticle']:
                article_info = extract_article_info(paper)
                if article_info and article_info['abstract']:
                    all_articles.append(article_info)

        time.sleep(1)

    if all_articles:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)

        print(f"成功下载 {len(all_articles)} 篇文章到 {output_json_file}")
        print(f"总耗时: {time.time() - start_time:.2f} 秒")
    else:
        print("未找到符合条件的文章")

def get_args():
    parser = ArgumentParser(description='Download PubMed articles within a specified date range')

    # 设置默认参数值 [2,5](@ref)
    parser.add_argument(
        "--output_json",
        type=str,
        default="./pubmed_articles.json",  # 默认输出路径
        help="Path to the JSON output file (default: ./pubmed_articles.json)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2020/11/01",  # 默认开始日期
        help="Start date in YYYY/MM/DD format (default: 2023/11/01)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024/11/30",  # 默认结束日期
        help="End date in YYYY/MM/DD format (default: 2023/11/30)",
    )
    parser.add_argument(
        "--num_articles",
        type=int,
        default=500,  # 默认文章数量
        help="Maximum number of articles to download (default: 500)",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    get_pubmed_data(
        output_json_file=args.output_json,
        start_date=args.start_date,
        end_date=args.end_date,
        max_num_articles=args.num_articles
    )