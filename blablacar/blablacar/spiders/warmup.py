import scrapy
import persistent
from datetime import datetime
MEMBER_PREFIX = 'https://www.blablacar.fr/membre/profil/'


class WarmupSpider(scrapy.Spider):
    name = "warmup"
    allowed_domains = ["blablacar.fr"]
    to_process_next  = set()
    # start_urls = ['https://www.blablacar.fr/membre/profil/tO9TUCYttaFHA0ixh6_Rcw']

    def start_requests(self):
        urls = [MEMBER_PREFIX+id_ for id_ in persistent('to_process.my')]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
        # TODO add hostname
        print('done with `to_process`')
        persistent.save_var('to_process_next.my', self.to_process_next)

    def parse(self, response):
        scraptime = datetime.utcnow()
        # user
        #  info
        # TODO: don't fail on that https://www.blablacar.fr/membre/profil/tO9TUCYttaFHA0ixh6_Rcw?page=2
        id_ = response.url.split('/')[-1].split('?')[0]
        name = response.css('h1.ProfileCard-info--name::text').extract_first().strip().encode('utf-8')
        age = int(response.css('div.ProfileCard-info:nth-child(2)::text').extract_first().strip().split()[0])
        profile_info = response.css('div.ProfileCard').css('div.ProfileCard-row')
        experience = profile_info[0].css('span.megatip-popover::text').extract_first().strip().encode('utf-8')
        # might although be title instead of oldtitle
        prefs = [_.strip().encode('utf-8') for _ in profile_info[2].xpath('span[contains(@class, "big-prefs")]/@oldtitle').extract()]
        bio_raw = response.xpath('//div[contains(@class, "member-bio")]/p/text()').extract()
        bio = ''.join((_.strip().encode('utf-8').replace('\n', ' ') for _ in bio_raw))
        #  activity
        activity = response.css('div.main-column-block:nth-child(2)')
        text_activity = [':'.join(_.strip().split(':')[1:]).strip().encode('utf-8')
                         for _ in activity.xpath('ul/li/text()').extract()]
        num_post = int(text_activity[0])
        reply_rate = int(text_activity[1][:-1])
        # check if dateparser supports French…
        last_ping = int(scraptime - text_activity[2])
        joined = text_activity[3]

        # or (old)title for text one (but probably I can check later manually)
        verif = [int(_) for _ in response.css('ul.verification-list').xpath('li/span/@data-hasqtip').extract()]

        # car
        car = response.css('ul.user-car-details li')
        car_name = car[0].xpath('h4/strong/text()').extract_first().strip().encode('utf-8')
        car_color, car_comfort = [_.strip().split(':')[-1]).strip().encode('utf-8') for _ in car[1:].xpath('text()').extract()]
        car_comfort_num = [int(i.split('_')[-1]) for i in car[0].xpath('h4/span/@class').extract_first().split() if 'star' in i][0]

        reviews_div = response.css('div.user-comments-container')
        for r in reviews_div.xpath('div/ul/li/article'):
            from_ = r.xpath('a/@href').extract_first().split('/')[-1].encode('ascii')
            to_ = id_
            text_raw = r.xpath('div[contains(@class, "Speech-content")]/p/text()').extract()
            text = ''.join((_.strip().encode('utf-8').replace('\n', ' ') for _ in text_raw))
            grade_class = r.xpath('div[contains(@class, "Speech-content")]/h3/@class')
            grade  = [int(i.split('--')[-1]) for i in grade_class.extract_first().split() if '--' in i][0]
            when = parse_date(r.xpath('footer/time/@datetime').extract_first())
        next_link = reviews_div.xpath('div[contains(@class, "pagination")]/ul/li[contains(@class, "next")]/a/@href').extract_first()
