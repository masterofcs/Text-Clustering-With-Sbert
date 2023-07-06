import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from news.items import NewsItem
from bs4 import BeautifulSoup
class TuoitreSpider(CrawlSpider):
    name = "tuoitre"
    allowed_domains = ["tuoitre.vn"]
    start_urls = ["https://tuoitre.vn"]

    rules = [
        Rule(LinkExtractor(restrict_css=(".box-category-link-title"))),
        Rule(LinkExtractor(restrict_css=(".detail__section")), callback="parse_item"),
    ]
    def parse_item(self, response):
        title = response.css('.detail__main')
        item = NewsItem()
        item["category"] = title.css('.detail-cate a::text').get().strip(),
        item["publishdate"] = title.css('.detail-time div::text').get().strip(),
        item["title"] = title.css('.detail-title.article-title::text').get().strip(),
        item["author"] = title.css('.author-info a.name::text').get().strip(),
        item["content"] = BeautifulSoup(title.css('.detail-cmain').get()).getText().strip()
        print(item)
        return item
        # for next_page in response.css('a.next'):
        #     yield
