from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import io
from PIL import Image
import os
import re
import utils
import numpy as np
from skimage import io as skio
import warnings

CHROME_DRIVER_PATH = "chromedriver"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
SCALE = 2.25

HTML_DIR = "html"
SCREENSHOTS_DIR = "screenshots"

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

def get_js_elements():
    return "var elements = Array.from(document.getElementsByTagName('*'));"

def get_js_cls_pos():
    return "var cls_pos = elements.map(element => [element.className, element.getBoundingClientRect()]); return cls_pos;"

def browse_urls(driver, url_list):
    for count, url in enumerate(url_list):
        driver.get(url)
        elements = driver.execute_script(get_js_elements() + get_js_cls_pos())
        #print (elements)
        total = 0
        for i in range(len(elements)):
            element = elements[i]
            class_name = element[0]
            if class_name == '':
                continue
            total += 1
            #print (total, class_name)

            top = max(0, int(element[1]['top']))
            bottom = min(int(element[1]['bottom']),WINDOW_HEIGHT-1)
            left = max(int(element[1]['left']),0)
            right = min(int(element[1]['right']), WINDOW_WIDTH-1)

            #print ('top', top, 'bottom', bottom, 'left', left, 'right', right)

            MASK_DIR = './' + SCREENSHOTS_DIR + '/masks/' + str(count+1) + '/'
            if not os.path.exists(MASK_DIR):
                os.makedirs(MASK_DIR)

            mask = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
            mask[top:bottom+1,left:right+1] = np.ones((bottom-top+1, right-left+1))

            skio.imsave(MASK_DIR + str(total) + '_' + class_name.replace(' ', '.') + '.png', mask)
        
        IMG_DIR = './' + SCREENSHOTS_DIR + '/images/' + str(count+1) + '/'
        if not os.path.exists(IMG_DIR):
            os.makedirs(IMG_DIR)

        screenshot = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot))
        skio.imsave(IMG_DIR + str(count+1) + '.png', img)

        '''
        width, height = img.size
        print ('width', width, 'height', height)
        resized_img = img.resize((WINDOW_WIDTH, WINDOW_HEIGHT))
        resized_img.save("{}/{}.jpg".format(SCREENSHOTS_DIR, count + 1), "JPEG")
        '''

@utils.func_timing_long(utils.print_out)
def take_browser_screenshots(url_list):
    chrome_options = Options()
    chrome_options.add_argument("log-level=2")
    chrome_options.add_argument("headless")

    driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH,
                              chrome_options=chrome_options)
    driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    if not os.path.exists(SCREENSHOTS_DIR):
        os.makedirs(SCREENSHOTS_DIR)

    try:
        browse_urls(driver, url_list)
    finally:
        driver.quit()

def get_url_list(html_dir):
    files_list = sorted(os.listdir(html_dir),
                        key=lambda x: (int(re.sub("\D", "", x)), x))

    pwd_path = os.path.dirname(os.path.realpath(__file__))
    pwd_path = pwd_path.replace(os.path.sep, "/")

    url_list = []
    for file_name in files_list:
        url = "file://{}/{}/{}".format(pwd_path, html_dir, file_name)
        url_list.append(url)

    return url_list

def main():
    url_list = get_url_list(HTML_DIR)
    print("len(url_list):", len(url_list))
    take_browser_screenshots(url_list)

if __name__ == "__main__":
    main()