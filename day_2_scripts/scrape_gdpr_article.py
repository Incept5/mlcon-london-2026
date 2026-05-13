import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

HERE = Path(__file__).parent


def scrape_gdpr_article(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        content_selectors = [
            'div[data-elementor-type="single-post"]',
            '.entry-content',
            '.post-content',
            'main',
            'article',
            '.content',
            'body'
        ]

        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                print(f"Found content using selector: {selector}")
                break

        if not main_content:
            print("Could not find main content container, using entire body")
            main_content = soup.find('body')

        if main_content:
            text_content = []

            title = soup.find('title')
            if title:
                title_text = title.get_text().strip()
                text_content.append(f"# {title_text}\n")

            # GDPR-specific: match headings like "Article 17"
            article_heading = soup.find(['h1', 'h2'], string=re.compile(r'Article \d+', re.IGNORECASE))
            if article_heading:
                text_content.append(f"## {article_heading.get_text().strip()}\n")

            content_elements = main_content.find_all([
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'p', 'li',
                'div', 'span',
                'td', 'th'
            ])

            filtered_elements = []
            for element in content_elements:
                if element.find_parent(['nav', 'footer', 'header']):
                    continue

                element_classes = element.get('class', [])
                skip_classes = ['nav', 'menu', 'footer', 'sidebar', 'widget', 'meta']
                if any(skip_class in ' '.join(element_classes).lower() for skip_class in skip_classes):
                    continue

                text = element.get_text().strip()
                if not text or len(text) < 3:
                    continue

                if len(text) == 1:
                    continue

                filtered_elements.append((element, text))

            processed_texts = set()
            for element, text in filtered_elements:

                if text in processed_texts:
                    continue
                processed_texts.add(text)

                if element.name and element.name.startswith('h'):
                    level = int(element.name[1]) if element.name[1].isdigit() else 2
                    heading_prefix = '#' * min(level, 6)
                    text_content.append(f"\n{heading_prefix} {text}\n")
                elif element.name == 'li':
                    text_content.append(f"- {text}")
                elif element.name in ['td', 'th']:
                    # Table cells handled separately below to preserve structure
                    if element.parent and element.parent.name == 'tr':
                        continue
                else:
                    text_content.append(text)

            tables = main_content.find_all('table')
            for table in tables:
                text_content.append("\n### Table\n")
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_text = ' | '.join(cell.get_text().strip() for cell in cells)
                        if row_text.strip():
                            text_content.append(f"| {row_text} |")
                text_content.append("")

            final_text = '\n'.join(text_content)

            final_text = re.sub(r'\n{4,}', '\n\n\n', final_text)
            final_text = re.sub(r' {2,}', ' ', final_text)
            final_text = re.sub(r'^\s+', '', final_text, flags=re.MULTILINE)
            final_text = final_text.strip()

            output_path = HERE / "data" / "gdpr_article_content.txt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_text)

            print(f"Successfully scraped content and saved to {output_path}")
            return final_text

        else:
            print("Could not find any content on the page.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error occurred during scraping: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def scrape_multiple_gdpr_articles(base_url_pattern, article_numbers):
    all_content = {}

    for article_num in article_numbers:
        url = base_url_pattern.format(article_num)
        print(f"\nScraping Article {article_num} from {url}")

        content = scrape_gdpr_article(url)
        if content:
            all_content[article_num] = content
            print(f"Successfully scraped Article {article_num}")
        else:
            print(f"Failed to scrape Article {article_num}")

    return all_content


if __name__ == "__main__":
    url = "https://gdpr-info.eu/art-17-gdpr/"
    content = scrape_gdpr_article(url)

    if content:
        print("\n" + "=" * 50)
        print("Preview of scraped content:")
        print("=" * 50)
        print(content[:500] + "..." if len(content) > 500 else content)
        print(f"\nTotal characters scraped: {len(content)}")
