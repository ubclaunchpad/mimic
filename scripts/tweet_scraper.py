"""Script to scrape tweets from the advanced search Twitter page to get over the 3200 tweets limit"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from bs4 import BeautifulSoup as bs
from datetime import date, timedelta
import time
import sys
import io
import re
import zipfile

def main():
    # Trump's Presidential Campaign 2016-06-15 - present
    # The timeline can be toggled below
    from_date = date(2016,6,15)
    to_date = date(2019,2,24)
    # Scraping tweets every "gap" days
    gap = 5
    days = (to_date-from_date).days

    # HTML contents will be appended here
    all_browser = ""

    # Chrome is used
    browser = webdriver.Chrome()
    wait = WebDriverWait(browser, 2)

    # Launches Twitter advanced search page
    browser.get('https://twitter.com/search-advanced?lang=en&lang=en&lang=en&lang=en&lang=en')

    # Iterates through desired dates to obtain tweets
    for day in range(0,days,gap):
        from_ = from_date + timedelta(days=day)
        to_ = from_date + timedelta(days=day+gap)
        from_input = "{dt.year}-{dt.month}-{dt.day}".format(dt = from_)
        to_input = "{dt.year}-{dt.month}-{dt.day}".format(dt = to_)
        time.sleep(2)

        try:
            user_field = browser.find_element_by_xpath("//input[@type='text' and @name='from']")
            user_field.send_keys("realDonaldTrump")

            from_field = browser.find_element_by_xpath("//input[contains(@class, 'input-sm') and @name='since']")
            from_field.send_keys(from_input)
            to_field = browser.find_element_by_xpath("//input[contains(@class, 'input-sm') and @name='until']")
            to_field.send_keys(to_input)

            search_button = browser.find_element_by_xpath("//button[@type='submit' and contains(@class, 'EdgeButton')]")
            search_button.click()

            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "tweet-text")))
                scroller(browser, wait)
            except TimeoutException:
                pass

            all_browser += browser.page_source

            browser.execute_script("window.history.go(-1)")
        except:
            # Returns to original search page
            browser.get('https://twitter.com/search-advanced?lang=en&lang=en&lang=en&lang=en&lang=en')

    # Parses out the individual tweets from HTML
    tweets = ""
    for page in all_browser.split("<!DOCTYPE html>"):
        soup = bs(page, "lxml")
        for tweet in soup.find_all(class_="tweet-text", text=True):
            tweets += tweet.text + "\n\n"

    tweets = re.sub("\\npic.twitter.*\\n", "", tweets)

    # Size of HTML scraped
    print("HTML size: {} MB".format(sys.getsizeof(all_browser)/1e6))

    # Approximately number of words and size of tweets
    print("Words: {}\nTweets Size: {} MB".format(sys.getsizeof(tweets)/5, sys.getsizeof(tweets)/1e6))

    # Saves tweets as zip
    mf = io.BytesIO()
    with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('trump_tweets.txt', str.encode(tweets, 'utf-8'))
    with open("../data/trump_tweets.zip", "wb") as f:  # use `wb` mode
        f.write(mf.getvalue())

class last_element_is_the_same():
    """
    Class used to detect when end of page is reached

    Takes in a tuple of (HTML attribute, name) and text of previous tweet
    """
    def __init__(self, locator, previous):
        # previous is the last tweet before scrolling in text form
        self.locator = locator
        self.previous = previous
    def __call__(self, browser):
        new_tweets = browser.find_elements(*self.locator)
        if new_tweets[-1].text != self.previous:
            return True
        else:
            return False

def scroller(browser, wait):
    """
    Scrolls to end of page.

    Takes in the browser driven and the 'WebDriverWait' object
    """
    while True:
        tweets = browser.find_elements_by_class_name("tweet-text")
        browser.execute_script("arguments[0].scrollIntoView();", tweets[-1])
        try:
            wait.until(last_element_is_the_same((By.CLASS_NAME, "tweet-text"), tweets[-1].text))
        except TimeoutException:
            break

if __name__ == "__main__":
    main()
