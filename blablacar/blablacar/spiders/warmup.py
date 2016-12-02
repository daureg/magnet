# -*- coding: utf-8 -*-
import scrapy
import persistent
MEMBER_PREFIX = 'https://www.blablacar.fr/membre/profil/'


class WarmupSpider(scrapy.Spider):
    name = "warmup"
    allowed_domains = ["blablacar.fr"]
    to_process_next  = set()
    start_urls = ['https://www.blablacar.fr/membre/profil/tO9TUCYttaFHA0ixh6_Rcw']

    def start_requests(self):
            urls = [MEMBER_PREFIX+id_ for id_ in persistent('to_process.my')]
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)
            # TODO add hostname
            print('done with `to_process`')
            persistent.save_var('to_process_next.my', self.to_process_next)

    def parse(self, response):
        name = response.css('h1.ProfileCard-info--name::text').extract_first().strip().encode('utf-8')
        age = int(response.css('div.ProfileCard-info:nth-child(2)::text').extract_first().strip().split()[0]) 
        pass
