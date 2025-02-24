import json
import logging
from datetime import datetime
from notion_client import Client
from pathlib import Path
import os
from notion_converter import NotionConverter

# 优先使用环境变量，如果环境变量不存在则从 config.py 导入
NOTION_TOKEN = os.environ.get('NOTION_TOKEN')
DATABASE_ID = os.environ.get('DATABASE_ID')

if not NOTION_TOKEN or not DATABASE_ID:
    try:
        from config import NOTION_TOKEN, DATABASE_ID
    except ImportError:
        raise ValueError("请确保设置了 NOTION_TOKEN 和 DATABASE_ID 环境变量或创建了 config.py 文件")


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class NotionToHugo:
    def __init__(self):
        self.notion = Client(auth=NOTION_TOKEN)
        self.converter = NotionConverter(self.notion)
        self.posts_dir = Path("content/posts")

    def get_database_entries(self):
        """获取Notion数据库中的所有条目"""
        try:
            response = self.notion.databases.query(database_id=DATABASE_ID)
            return response.get("results", [])
        except Exception as e:
            logging.error(f"获取数据库条目时出错: {e}", exc_info=True)
            return []

    def check_post_needs_update(self, entry, file_path):
        """检查文章是否需要更新"""
        if not file_path.exists():
            return True

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                try:
                    # 解析front matter
                    front_matter_str = content.split("---")[1]
                    front_matter = json.loads(front_matter_str)
                    # 获取生成时间
                    generated_time = front_matter.get("generated_time", "")
                    # 获取Notion的最后修改时间
                    last_edited_time = entry.get("last_edited_time", "")
                    # 如果文章的生成时间晚于Notion的最后修改时间，则不需要更新
                    if generated_time and last_edited_time:
                        if generated_time > last_edited_time:
                            return False
                except (IndexError, json.JSONDecodeError, KeyError):
                    pass
        except Exception as e:
            logging.error(f"检查文章更新状态时出错: {e}", exc_info=True)
            return True

        return True

    def create_hugo_post(self, entry):
        """创建Hugo文章文件"""
        try:
            # 获取文章标题用于生成文件名
            properties = entry.get("properties", {})
            title = properties.get("Name", {}).get("title", [])
            title = title[0].get("plain_text", "") if title else "Untitled"
            slug = title.lower().replace(" ", "-")
            file_name = f"{slug}.md"

            # 确保posts目录存在
            self.posts_dir.mkdir(parents=True, exist_ok=True)

            # 获取文件路径
            file_path = self.posts_dir / file_name

            # 检查文章是否需要更新
            if not self.check_post_needs_update(entry, file_path):
                logging.info(f"跳过未修改的文章: {file_name}")
                return

            # 只有需要更新时才执行转换
            post_data = self.converter.convert_to_markdown(entry)
            if not post_data:
                return

            # 添加生成时间到front matter
            post_data["front_matter"]["generated_time"] = datetime.utcnow().isoformat()

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write(json.dumps(post_data["front_matter"], ensure_ascii=False, indent=2))
                f.write("\n---\n\n")
                f.write(post_data["content"])

            logging.info(f"成功创建文章: {file_name}")

        except Exception as e:
            logging.error(f"创建文章时出错: {e}", exc_info=True)

    def sync_all_posts(self):
        """同步所有文章"""
        # 先同步数据库中的文章
        entries = self.get_database_entries()
        for entry in entries:
            if entry.get("properties")['draft']['checkbox']:
                continue
            self.create_hugo_post(entry)
        
        # 清理已删除的文章，传入已获取的entries
        self.clean_deleted_posts(entries)

    def clean_deleted_posts(self, entries):
        """清理本地已删除或已变为草稿的文章"""
        try:
            # 获取数据库中所有非草稿文章的文件名列表
            db_filenames = set()
            for entry in entries:
                if entry.get("properties")['draft']['checkbox']:
                    continue
                # 直接从entry中获取文件名，避免完整的markdown转换
                properties = entry.get("properties", {})
                title = properties.get("Name", {}).get("title", [])
                title = title[0].get("plain_text", "") if title else "Untitled"
                slug = title.lower().replace(" ", "-")
                file_name = f"{slug}.md"
                db_filenames.add(file_name)

            # 获取本地文章文件列表
            local_files = set(f.name for f in self.posts_dir.glob("*.md"))

            # 找出需要删除的文件
            files_to_delete = local_files - db_filenames

            # 删除文件
            for filename in files_to_delete:
                file_path = self.posts_dir / filename
                file_path.unlink()
                logging.info(f"删除文章: {filename}")

        except Exception as e:
            logging.error(f"清理文章时出错: {e}", exc_info=True)

def main():
    converter = NotionToHugo()
    converter.sync_all_posts()

if __name__ == "__main__":
    main()