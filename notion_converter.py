import json
import logging
from datetime import datetime


class NotionConverter:
    def __init__(self, notion_client):
        self.notion = notion_client
        self.custom_transformers = {}

    def set_custom_transformer(self, block_type, transformer):
        """设置自定义块转换器"""
        self.custom_transformers[block_type] = transformer
        return self

    def convert_to_markdown(self, entry):
        """将单个Notion条目转换为Hugo文章格式"""
        try:
            # 提取文章属性
            properties = entry.get("properties", {})

            # 获取标题
            title = properties.get("Name", {}).get("title", [])
            title = title[0].get("plain_text", "") if title else "Untitled"

            # 获取标签
            tags = [
                tag.get("name", "")
                for tag in properties.get("tags", {}).get("multi_select", [])
            ]

            # 获取分类
            categories = [
                cat.get("name", "")
                for cat in properties.get("categories", {}).get("multi_select", [])
            ]

            # 获取摘要
            summary = properties.get("summary", {}).get("rich_text", [])
            summary = summary[0].get("plain_text", "") if summary else ""

            # 获取自定义front-matter
            custom_front_matter = properties.get("custom-front-matter", {}).get(
                "rich_text", []
            )
            custom_front_matter = (
                custom_front_matter[0].get("plain_text", "")
                if custom_front_matter
                else ""
            )

            # 获取作者
            authors = [
                author.get("name", "")
                for author in properties.get("authors", {}).get("multi_select", [])
            ]

            # 获取创建时间
            created_time = properties.get("Created time", {}).get("created_time", "")
            date = datetime.fromisoformat(created_time.replace("Z", "+00:00")).strftime(
                "%Y-%m-%d"
            )

            # 获取内容
            content = self._get_page_content(entry["id"])

            # 创建front matter
            front_matter = {
                "title": title,
                "date": date,
                "tags": tags,
                "categories": categories,
                "summary": summary,
                "authors": authors,
                "draft": False,
            }

            # 如果有自定义front matter，解析并添加
            if custom_front_matter:
                try:
                    custom_data = json.loads(custom_front_matter)
                    front_matter.update(custom_data)
                except json.JSONDecodeError:
                    logging.warning(f"自定义front matter格式错误: {custom_front_matter}")

            # 生成文章文件名
            slug = title.lower().replace(" ", "-")
            file_name = f"{slug}.md"

            return {
                "file_name": file_name,
                "front_matter": front_matter,
                "content": content,
            }

        except Exception as e:
            logging.error(f"处理条目时出错: {e}", exc_info=True)
            return None

    def _get_page_content(self, page_id):
        """获取Notion页面的内容"""
        try:
            blocks = self.notion.blocks.children.list(block_id=page_id).get(
                "results", []
            )
            return self._blocks_to_markdown(blocks)
        except Exception as e:
            logging.error(f"获取页面内容时出错: {e}", exc_info=True)
            return ""

    def _blocks_to_markdown(self, blocks, nesting_level=0):
        """将Notion块转换为Markdown格式"""
        content = ""
        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            # 检查是否有自定义转换器
            if block_type in self.custom_transformers:
                content += self.custom_transformers[block_type](block, nesting_level)
                continue

            # 默认转换器
            if block_type == "paragraph":
                content += (
                    self._convert_rich_text(block_data.get("rich_text", [])) + "\n\n"
                )
            elif block_type == "heading_1":
                content += (
                    f"# {self._convert_rich_text(block_data.get('rich_text', []))}\n\n"
                )
            elif block_type == "heading_2":
                content += (
                    f"## {self._convert_rich_text(block_data.get('rich_text', []))}\n\n"
                )
            elif block_type == "heading_3":
                content += f"### {self._convert_rich_text(block_data.get('rich_text', []))}\n\n"
            elif block_type == "bulleted_list_item":
                content += f"{' ' * nesting_level * 2}- {self._convert_rich_text(block_data.get('rich_text', []))}\n"
            elif block_type == "numbered_list_item":
                content += f"{' ' * nesting_level * 2}1. {self._convert_rich_text(block_data.get('rich_text', []))}\n"
            elif block_type == "to_do":
                checked = "x" if block_data.get("checked", False) else " "
                content += f"{' ' * nesting_level * 2}- [{checked}] {self._convert_rich_text(block_data.get('rich_text', []))}\n"
            elif block_type == "code":
                language = block_data.get("language", "")
                code = self._convert_rich_text(block_data.get("rich_text", []))
                content += f"```{language}\n{code}\n```\n\n"
            elif block_type == "quote":
                content += (
                    f"> {self._convert_rich_text(block_data.get('rich_text', []))}\n\n"
                )
            elif block_type == "divider":
                content += "---\n\n"
            elif block_type == "callout":
                emoji = block_data.get("icon", {}).get("emoji", "")
                text = self._convert_rich_text(block_data.get("rich_text", []))
                content += f"> {emoji} {text}\n\n"
            elif block_type == "image":
                image_url = block_data.get("file", {}).get("url", "")
                caption = self._convert_rich_text(block_data.get("caption", []))
                if image_url:
                    # 下载图片并保存到本地
                    try:
                        import os
                        from pathlib import Path

                        import requests

                        # 创建static/images目录
                        image_dir = Path("static/images")
                        image_dir.mkdir(parents=True, exist_ok=True)

                        # 生成图片文件名
                        image_filename = f"notion_{block['id']}{image_url.split('/')[4]}{os.path.splitext(image_url)[1].split('?')[0]}"
                        image_path = image_dir / image_filename

                        # 下载图片
                        response = requests.get(image_url)
                        response.raise_for_status()

                        # 保存图片
                        with open(image_path, "wb") as f:
                            f.write(response.content)

                        # 使用相对路径在Markdown中引用图片
                        content += f"![{caption}](/images/{image_filename})\n\n"
                    except Exception as e:
                        logging.error(f"下载图片时出错: {e}", exc_info=True)
                        content += f"![{caption}]({image_url})\n\n"

            # 处理子块
            if block.get("has_children", False):
                child_blocks = self.notion.blocks.children.list(
                    block_id=block["id"]
                ).get("results", [])
                content += self._blocks_to_markdown(child_blocks, nesting_level + 1)

        return content

    def _convert_rich_text(self, rich_text):
        """转换富文本格式"""
        text = ""
        for rt in rich_text:
            content = rt.get("plain_text", "")
            annotations = rt.get("annotations", {})

            # 应用文本格式
            if annotations.get("bold"):
                content = f"**{content}**"
            if annotations.get("italic"):
                content = f"_{content}_"
            if annotations.get("strikethrough"):
                content = f"~~{content}~~"
            if annotations.get("code"):
                content = f"`{content}`"

            # 处理链接
            if rt.get("href"):
                content = f"[{content}]({rt['href']})"

            text += content

        return text
