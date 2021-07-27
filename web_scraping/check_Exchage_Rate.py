import requests
from bs4 import BeautifulSoup
import time
import smtplib


class Currency:
    DOLLAR_RUB = 'https://www.google.ru/search?q=%D0%BA%D1%83%D1%80%D1%81+%D0%B4%D0%BE%D0%BB%D0%BB%D0%B0%D1%80%D0%B0&newwindow=1&sxsrf=ALeKk00Ane3coeHbF_vSCYEJGyH1W2l6Mg%3A1626880704678&source=hp&ei=wDr4YMPEJqytrgShh4rgBA&iflsig=AINFCbYAAAAAYPhI0C0zBFG6wf4dHpByy3w2ZNdw3lDg&oq=%D0%9A%D1%83%D1%80%D1%81+&gs_lcp=Cgdnd3Mtd2l6EAEYATIECCMQJzIECCMQJzIECCMQJzIICAAQsQMQgwEyAggAMggIABCxAxCDATICCAAyAggAMggIABCxAxCDATIFCAAQsQM6BwgjEOoCECc6CAguELEDEIMBOgsILhCxAxDHARCjAjoCCC46DgguELEDEMcBEKMCEJMCOgUILhCxAzoHCCMQsQIQJzoKCAAQsQMQgwEQCjoECAAQCjoHCAAQsQMQCjoICAAQChABECo6BggAEAoQAToOCC4QsQMQgwEQxwEQowJQphJY_TVgvEdoB3AAeAGAAb0DiAGxEZIBCTIuNy4zLjAuMZgBAKABAaoBB2d3cy13aXqwAQo&sclient=gws-wiz'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36'}
    current_converted_price = 0
    differense = 5

    def __init__(self):
        self.current_converted_price = float(self.get_current_price().replace(",", '.'))

    def get_current_price(self):
        full_page = requests.get(self.DOLLAR_RUB, headers=self.headers)
        soup = BeautifulSoup(full_page.content, 'html.parser')
        convert = soup.findAll("span", {"class": "DFlfde", "class": "SwHCTb", "data-precision": 2})
        return convert[0].text

    def check_currency(self):
        currency = float(self.get_current_price().replace(",", '.'))
        if currency >= self.current_converted_price + self.differense:
            print("Курс сильно вырос, может пора что-то делать?")
            self.send_mail()
        elif currency <= self.current_converted_price  - self.differense:
            print("Курс сильно упал, может пора что-то делать?")
            self.send_mail()
        print("Сейчас курс: 1 доллар = ", str(currency))
        time.sleep(3)
        self.check_currency()
    def send_mail(self):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login('danil.isacin.78@gmail.com', 'brkzctxywalnqtte')
        subject = 'Exchange Rates'
        body = 'dollar exchange rate has changed!'
        message = f'Subject: {subject}\n\n{body}'
        server.sendmail(
            'admin@itproger.com',
            'danil.isacin.78@gmail.com',
            message
        )
        server.quit()

currency = Currency()
currency.check_currency()
currency.send_mail()

