# vim: set fileencoding=utf-8
import scrapy
import persistent
import logging
from datetime import datetime
from dateparser import parse as parse_date
from urlparse import urlparse
LANGUAGE = 'fr'
WEBSITE = {'fr': 'https://www.blablacar.fr',
           'en': 'https://www.blablacar.co.uk',
           }[LANGUAGE]
MEMBER_PREFIX = WEBSITE + {'fr': '/membre/profil/',
                           'en': '/user/show/',
                           }[LANGUAGE]
ACTIVITIES_STR = {'fr': {'post': 'Annonces', 'rate': 'Taux',
                         'last': 'Dernière', 'member': 'Membre'},
                  'en': {'post': 'rides', 'rate': 'message',
                         'last': 'Last', 'member': 'Member'},
                  }[LANGUAGE]


def parse_activity(activity, scraptime):
    activities = [_.strip().encode('utf-8')
                  for _ in activity.xpath('ul/li/text()').extract()]
    text_activity = dict()
    for line in activities:
        first_word = line.split()[0]
        content = ':'.join(line.split(':')[1:]).strip()
        text_activity[first_word] = content
    num_post, reply_rate, last_ping, joined = 4*[None, ]
    if 'Annonces' in text_activity:
        num_post = int(text_activity['Annonces'])
    if 'Taux' in text_activity:
        reply_rate = int(text_activity['Taux'][:-1])
    # for https://github.com/scrapinghub/dateparser/issues/268
    last_ping = text_activity['Dernière'].replace('mar.', '')
    last_ping = int((scraptime - parse_date(last_ping)).total_seconds())
    joined = parse_date(text_activity['Membre']).date()
    return num_post, reply_rate, last_ping, joined


class WarmupSpider(scrapy.Spider):
    name = "warmup"
    allowed_domains = [urlparse(WEBSITE).netloc[4:]]
    seen_users = set()
    all_reviews = dict()

    def start_requests(self):
        self.seen_users = persistent.load_var('seen_users.my')
        urls = [MEMBER_PREFIX+id_ for id_ in persistent.load_var('next_users.my')]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # user
        #  info
        id_ = response.url.split('/')[-1].split('?')[0]
        if id_ in self.seen_users:
            return
        if id_ not in self.all_reviews:
            self.all_reviews[id_] = list()
        scraptime = datetime.utcnow()
        name = response.css('h1.ProfileCard-info--name::text').extract_first().strip().encode('utf-8')
        age = int(response.css('div.ProfileCard-info:nth-child(2)::text').extract_first().strip().split()[0])
        profile_info = response.css('div.ProfileCard').css('div.ProfileCard-row')
        experience = profile_info[0].css('span.megatip-popover::text').extract_first()
        if experience is not None:
            experience = experience.strip().encode('utf-8')
        # might although be oldtitle instead of title
        prefs = [_.strip().encode('utf-8') for _ in profile_info[-1].xpath('span[contains(@class, "big-prefs")]/@title').extract()]
        bio_raw = response.xpath('//div[contains(@class, "member-bio")]/p/text()').extract()
        bio = ''.join((_.strip().encode('utf-8').replace('\n', ' ') for _ in bio_raw)).strip('""')
        #  activity
        num_post, reply_rate, last_ping, joined = parse_activity(response.css('div.main-column-block:nth-child(2)'), scraptime)

        verif = [_.strip().encode('utf-8') for _ in response.css('ul.verification-list li span::text').extract()]

        # car
        car = response.css('ul.user-car-details li')
        if car:
            car_name = car[0].xpath('h4/strong/text()').extract_first().strip().encode('utf-8')
            car_color, car_comfort = [_.strip().split(':')[-1].strip().encode('utf-8')
                                      for _ in car[1:].xpath('text()').extract()]
            car_comfort_num = [int(i.split('_')[-1])
                               for i in car[0].xpath('h4/span/@class').extract_first().split()
                               if 'star' in i][0]

        reviews_div = response.css('div.user-comments-container')
        for r in reviews_div.xpath('div/ul/li/article'):
            from_ = r.xpath('a/@href').extract_first()
            if not from_:
                # a reviewer without profile, like Estelle here https://www.blablacar.fr/membre/profil/Hx6O3ju2jAXgT7MpEKijcg
                continue
            from_ = from_.split('/')[-1].encode('ascii')
            text_raw = r.xpath('div[contains(@class, "Speech-content")]/p/text()').extract()
            text = ''.join((_.strip().encode('utf-8').replace('\n', ' ') for _ in text_raw))
            grade_class = r.xpath('div[contains(@class, "Speech-content")]/h3/@class')
            grade = [int(i.split('--')[-1]) for i in grade_class.extract_first().split() if '--' in i][0]
            when = parse_date(r.xpath('footer/time/@datetime').extract_first())
            self.all_reviews[id_].append({'from': from_,
                                          'grade': grade,
                                          'text': text,
                                          'when': str(when),
                                          })

        next_link = reviews_div.xpath('div[contains(@class, "pagination")]/ul/li[contains(@class, "next")]/a/@href').extract_first()
        if next_link:
            yield scrapy.Request('{}{}'.format(WEBSITE, next_link), callback=self.parse)
        else:
            these_reviews = list(self.all_reviews[id_])
            del self.all_reviews[id_]
            logging.info('DONE_WITH {}'.format(id_))
            self.seen_users.add(id_)
            yield {'id': id_,
                   'name': name,
                   'age': age,
                   'experience': experience,
                   'preferences': prefs,
                   'biography': bio,
                   'num_posts': num_post,
                   'reply_rate': reply_rate,
                   'joined': str(joined),
                   'last_ping': last_ping,
                   'scraptime': str(scraptime),
                   'car': None if len(car) == 0 else {'name': car_name, 'color': car_color, 'comfort': car_comfort_num},
                   'verifications': verif,
                   'reviews': these_reviews,
                   }
