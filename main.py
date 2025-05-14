from text_analysis import analyze_sentiment_and_extract_kpis
import time
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import re
import streamlit as st
import traceback

def setup_driver():
    """Set up and return a Chrome webdriver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Add user agent to appear more like a regular browser
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Disable automation flags
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Set up driver
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Execute CDP commands to prevent detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        
        return driver
    except Exception as e:
        st.error(f"Error setting up Chrome driver: {str(e)}")
        raise e

def extract_reviews(url, debug_mode=False):
    driver = None
    reviews = []
    
    try:
        driver = setup_driver()
        # Load the hotel page
        driver.get(url)
        st.info("Accessing hotel page...")
        time.sleep(5)  # Increased wait time for page to load
        
        # Accept cookies if the dialog appears
        try:
            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, 
                    "//button[contains(@id, 'accept') or contains(text(), 'Accept') or contains(text(), 'Accepter')]"))
            )
            cookie_button.click()
            time.sleep(2)
        except TimeoutException:
            print("test")
        
        # Check if we need to navigate to the reviews section
        if "reviews" not in driver.current_url.lower():
            # Create a reviews-specific URL to directly access the reviews page
            base_url = url.split("?")[0].rstrip('/')
            reviews_url = f"{base_url}#tab-reviews"
            
            driver.get(reviews_url)
            st.info(f"Navigating directly to reviews page: {reviews_url}")
            time.sleep(5)  # Wait for page to load
        
        
        # First attempt: Try to find review cards using data-testid attribute (most reliable)
        review_cards = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="review-card"], div[data-testid="review-item"]')
        
        if not review_cards:
            st.info("No review cards found with data-testid, trying alternative selectors...")
            # Second attempt: Try alternative selectors commonly found on Booking.com
            review_cards = driver.find_elements(By.CSS_SELECTOR, '.c-review-block, .review_list_new_item_block, .review_item')
   

        if not review_cards:
            for i in range(3):  # Scroll a few times
                driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(2)
                
                # Try again after scrolling
                review_cards = driver.find_elements(By.CSS_SELECTOR, 
                    'div[data-testid="review-card"], div[data-testid="review-item"], .c-review-block, .review_list_new_item_block, .review_item')
                
                if review_cards:
                    break
        
        # Process each review card
        for card in review_cards:
            review_obj = {
                "reviewer_name": "Unknown",
                "reviewer_country": "Unknown",
                "review_date": "Unknown",
                "review_title": "No title",
                "review_score": "No score",
                "review_text": "No review text",
                "stay_info": "No stay information",
                "positive_comments": "",
                "negative_comments": ""
            }
            
            # Extract reviewer name
            try:
                # Try multiple selectors for reviewer name
                name_selectors = [
                    ".//span[@data-testid='review-title']",  # New data-testid approach
                    ".//div[@class='b08850ce41 f546354b44']",
                    ".//div[contains(@class, 'reviewer_name')]",
                    ".//div[contains(@class, 'bui-avatar-block__title')]"
                ]
                
                for selector in name_selectors:
                    try:
                        name_element = card.find_element(By.XPATH, selector)
                        if name_element.text.strip():
                            review_obj["reviewer_name"] = name_element.text.strip()
                            break
                    except:
                        continue
            except:
                pass
            
            # Extract reviewer country
            try:
                country_selectors = [
                    ".//span[@data-testid='review-country']",  # New data-testid approach
                    ".//div[@class='fff1944c52 fb14de7f14']/span",
                    ".//span[@class='d838fb5f41 aea5eccb71']",
                    ".//span[contains(@class, 'bui-avatar-block__subtitle')]"
                ]
                
                for selector in country_selectors:
                    try:
                        country_element = card.find_element(By.XPATH, selector)
                        if country_element.text.strip():
                            review_obj["reviewer_country"] = country_element.text.strip()
                            break
                    except:
                        continue
            except:
                pass
            
            # Extract review date
            try:
                date_selectors = [
                    ".//span[@data-testid='review-date']",  # New data-testid approach
                    ".//span[contains(@class, 'c-review-block__date')]",
                    ".//div[contains(@class, 'review_item_date')]"
                ]
                
                for selector in date_selectors:
                    try:
                        date_element = card.find_element(By.XPATH, selector)
                        if date_element.text.strip():
                            review_obj["review_date"] = date_element.text.strip()
                            break
                    except:
                        continue
            except:
                pass
            
            # Extract review score
            try:
                score_selectors = [
                    ".//div[@data-testid='review-score']",  # New data-testid approach
                    ".//div[@class='f63b14ab7a dff2e52086']",
                    ".//div[contains(@class, 'bui-review-score__badge')]",
                    ".//span[contains(@class, 'review-score-badge')]"
                ]
                
                for selector in score_selectors:
                    try:
                        score_element = card.find_element(By.XPATH, selector)
                        if score_element.text.strip():
                            review_obj["review_score"] = score_element.text.strip()
                            break
                    except:
                        continue
            except:
                pass
            
                # Extract positive comments - CRITICAL SECTION
            try:
                # First try with the exact CSS selector from your snippet
                try:
                    positive_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-positive"]')
                    if positive_element and positive_element.text.strip():
                        review_obj["positive_comments"] = positive_element.text.strip()
                except:
                    # Then try with data-testid="review-positive-text"
                    try:
                        positive_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-positive-text"]')
                        if positive_element and positive_element.text.strip():
                            review_obj["positive_comments"] = positive_element.text.strip()
                    except:
                        # Then try with span
                        try:
                            positive_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-positive-text"] span')
                            if positive_element and positive_element.text.strip():
                                review_obj["positive_comments"] = positive_element.text.strip()
                        except:
                            # Try with XPath - look for "Liked" label first
                            try:
                                liked_label = card.find_element(By.XPATH, './/div[contains(text(), "Liked:") or contains(text(), "J\'ai aimé:") or contains(text(), "Positif:")]')
                                if liked_label:
                                    # Get the sibling or next element which contains the actual text
                                    parent = liked_label.find_element(By.XPATH, './..')
                                    # Find all text elements within this parent section
                                    text_elements = parent.find_elements(By.XPATH, './/span | .//div[not(contains(text(), "Liked:") and not(contains(text(), "J\'ai aimé:")) and not(contains(text(), "Positif:"))]')
                                    
                                    positive_text = ""
                                    for elem in text_elements:
                                        if elem.text.strip() and "Liked:" not in elem.text and "J'ai aimé:" not in elem.text and "Positif:" not in elem.text:
                                            positive_text += elem.text.strip() + " "
                                    
                                    if positive_text.strip():
                                        review_obj["positive_comments"] = positive_text.strip()
                            except:
                                pass
            except:
                pass
            
            # Extract negative comments - CRITICAL SECTION
            try:
                # First try with the exact CSS selector from your snippet
                try:
                    negative_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-negative"]')
                    if negative_element and negative_element.text.strip():
                        review_obj["negative_comments"] = negative_element.text.strip()
                except:
                    # Then try with data-testid="review-negative-text"
                    try:
                        negative_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-negative-text"]')
                        if negative_element and negative_element.text.strip():
                            review_obj["negative_comments"] = negative_element.text.strip()
                    except:
                        # Then try with span
                        try:
                            negative_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-negative-text"] span')
                            if negative_element and negative_element.text.strip():
                                review_obj["negative_comments"] = negative_element.text.strip()
                        except:
                            # Try with XPath - look for "Disliked" label first
                            try:
                                disliked_label = card.find_element(By.XPATH, './/div[contains(text(), "Disliked:") or contains(text(), "Je n\'ai pas aimé:") or contains(text(), "Négatif:")]')
                                if disliked_label:
                                    # Get the sibling or next element which contains the actual text
                                    parent = disliked_label.find_element(By.XPATH, './..')
                                    # Find all text elements within this parent section
                                    text_elements = parent.find_elements(By.XPATH, './/span | .//div[not(contains(text(), "Disliked:") and not(contains(text(), "Je n\'ai pas aimé:")) and not(contains(text(), "Négatif:"))]')
                                    
                                    negative_text = ""
                                    for elem in text_elements:
                                        if elem.text.strip() and "Disliked:" not in elem.text and "Je n'ai pas aimé:" not in elem.text and "Négatif:" not in elem.text:
                                            negative_text += elem.text.strip() + " "
                                    
                                    if negative_text.strip():
                                        review_obj["negative_comments"] = negative_text.strip()
                            except:
                                pass
            except:
                pass
            
            # Extract review text - for cases where there's no specific positive/negative sections
            text_selectors = [
                ".//div[@data-testid='review-text']",
                ".//span[@data-testid='review-text']",
                ".//div[contains(@class, 'c-review__row')]//span",
                ".//div[contains(@class, 'review_item_review_content')]",
                ".//div[contains(@class, 'review_item_review')]//p"
            ]
            
            for selector in text_selectors:
                try:
                    elements = card.find_elements(By.XPATH, selector)
                    full_text = " ".join([el.text.strip() for el in elements if el.text.strip()])
                    if full_text:
                        review_obj["review_text"] = full_text
                        break
                except:
                    continue
            
            # If we have at least some valid data, add the review
            if (review_obj["positive_comments"] or review_obj["negative_comments"]):
                reviews.append(review_obj)
                if debug_mode:
                    st.info(f"Added review from {review_obj['reviewer_name']} with pos/neg comments")
        
        # If no reviews found via standard methods, try using the specific method from your snippet
        if not reviews:
            st.info("No reviews found with standard methods, trying direct approach...")
            
            try:
                # Direct approach from your snippet
                review_cards = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="review-card"]')
                
                if review_cards:
                    
                    for card in review_cards:
                        review_obj = {
                            "reviewer_name": "Unknown",
                            "reviewer_country": "Unknown",
                            "review_date": "Unknown",
                            "review_title": "No title",
                            "review_score": "No score",
                            "review_text": "No review text",
                            "stay_info": "No stay information",
                            "positive_comments": "",
                            "negative_comments": ""
                        }
                        
                        # Avis positif
                        try:
                            # First try with span
                            try:
                                positive_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-positive-text"] span')
                                if positive_element and positive_element.text.strip():
                                    review_obj["positive_comments"] = positive_element.text.strip()
                            except:
                                # Then try without span
                                try:
                                    positive_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-positive-text"]')
                                    if positive_element and positive_element.text.strip():
                                        review_obj["positive_comments"] = positive_element.text.strip()
                                except:
                                    pass
                        except:
                            pass
                            
                        # Avis négatif
                        try:
                            # First try with span
                            try:
                                negative_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-negative-text"] span')
                                if negative_element and negative_element.text.strip():
                                    review_obj["negative_comments"] = negative_element.text.strip()
                            except:
                                # Then try without span
                                try:
                                    negative_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-negative-text"]')
                                    if negative_element and negative_element.text.strip():
                                        review_obj["negative_comments"] = negative_element.text.strip()
                                except:
                                    pass
                        except:
                            pass
                        
                        # Add the review even if no positive/negative comments
                        # We'll populate other fields to make the review useful
                        try:
                            # Try to get reviewer name
                            try:
                                name_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-title"]')
                                if name_element and name_element.text.strip():
                                    review_obj["reviewer_name"] = name_element.text.strip()
                            except:
                                pass
                                
                            # Try to get score
                            try:
                                score_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-score"]')
                                if score_element and score_element.text.strip():
                                    review_obj["review_score"] = score_element.text.strip()
                            except:
                                pass
                                
                            # Try to get the entire review text if no specific pos/neg comments
                            if not review_obj["positive_comments"] and not review_obj["negative_comments"]:
                                try:
                                    text_element = card.find_element(By.CSS_SELECTOR, '[data-testid="review-text"]')
                                    if text_element and text_element.text.strip():
                                        full_text = text_element.text.strip()
                                        review_obj["review_text"] = full_text
                                        
                                        # Basic sentiment analysis to categorize as positive or negative
                                        try:
                                            score = float(review_obj["review_score"].replace(',', '.'))
                                            if score >= 7:  # Consider 7+ a positive review
                                                review_obj["positive_comments"] = full_text
                                            else:
                                                review_obj["negative_comments"] = full_text
                                        except:
                                            # If we can't parse the score, just put the text in both
                                            review_obj["positive_comments"] = full_text
                                            review_obj["negative_comments"] = full_text
                                except:
                                    pass
                        except:
                            pass
                        
                        # Always add the review to the list, even if incomplete
                        # This ensures we at least capture something from each review card
                        reviews.append(review_obj)
                        
                        if debug_mode:
                            st.info(f"Added review from {review_obj['reviewer_name']} (Score: {review_obj['review_score']})")
            except Exception as e:
                if debug_mode:
                    st.warning(f"Direct approach failed: {str(e)}")
                    st.error(traceback.format_exc())
        
        if not reviews and debug_mode:
            # Last resort: dump the entire page HTML for analysis
            page_source = driver.page_source
            with open("booking_page.html", "w", encoding="utf-8") as f:
                f.write(page_source)
            st.warning("No reviews extracted. Saved page HTML for analysis.")
            
        return reviews
        
    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"Error: {str(e)}")
        if debug_mode:
            st.error(f"Detailed error: {error_details}")
    finally:
        if driver:
            driver.quit()
    
    return reviews

def validate_booking_url(url):
    """Validate if the URL is a proper Booking.com hotel URL."""
    # Special case for paste URLs
    if url.startswith("https://pasteboard.co") or "paste" in url.lower():
        return True
        
    # Basic validation
    if not url.startswith("https://"):
        return False
    
    # Check if it's a Booking.com URL
    if not re.search(r"booking\.com", url):
        return False
    
    # Check if it's a hotel page
    if not re.search(r"hotel|hostel|apartment|resort", url):
        return False
        
    return True

def create_sentiment_dataframes(reviews):
    """Create separate dataframes for positive and negative comments."""
    positive_comments = []
    negative_comments = []
    
    for review in reviews:
        # Extract positive comments - check both positive_comments and review_text
        if review["positive_comments"]:
            positive_comments.append({
                "Reviewer": review["reviewer_name"],
                "Country": review["reviewer_country"],
                "Date": review["review_date"],
                "Score": review["review_score"],
                "Positive Comment": review["positive_comments"]
            })
        # If no specific positive comments but we have review text, check the score
        elif review["review_text"] != "No review text":
            try:
                score_str = review["review_score"].replace(',', '.')
                score = float(score_str)
                if score >= 7:  # Consider high scores as positive reviews
                    positive_comments.append({
                        "Reviewer": review["reviewer_name"],
                        "Country": review["reviewer_country"],
                        "Date": review["review_date"],
                        "Score": review["review_score"],
                        "Positive Comment": review["review_text"] + " (Auto-categorized based on score)"
                    })
            except:
                # If we can't parse the score or it's "No score", add to positive as a fallback
                if review["review_text"].strip():
                    positive_comments.append({
                        "Reviewer": review["reviewer_name"],
                        "Country": review["reviewer_country"],
                        "Date": review["review_date"],
                        "Score": review["review_score"],
                        "Positive Comment": review["review_text"] + " (Uncategorized review)"
                    })
        
        # Extract negative comments - check both negative_comments and review_text
        if review["negative_comments"]:
            negative_comments.append({
                "Reviewer": review["reviewer_name"],
                "Country": review["reviewer_country"],
                "Date": review["review_date"],
                "Score": review["review_score"],
                "Negative Comment": review["negative_comments"]
            })
        # If no specific negative comments but we have review text, check the score
        elif review["review_text"] != "No review text":
            try:
                score_str = review["review_score"].replace(',', '.')
                score = float(score_str)
                if score < 7:  # Consider low scores as negative reviews
                    negative_comments.append({
                        "Reviewer": review["reviewer_name"],
                        "Country": review["reviewer_country"],
                        "Date": review["review_date"],
                        "Score": review["review_score"],
                        "Negative Comment": review["review_text"] + " (Auto-categorized based on score)"
                    })
            except:
                # For uncategorized reviews that couldn't be added to positive, add here
                if review["review_text"].strip() and "Uncategorized" not in review["review_text"]:
                    negative_comments.append({
                        "Reviewer": review["reviewer_name"],
                        "Country": review["reviewer_country"],
                        "Date": review["review_date"],
                        "Score": review["review_score"],
                        "Negative Comment": review["review_text"] + " (Uncategorized review)"
                    })
    
    positive_df = pd.DataFrame(positive_comments)
    negative_df = pd.DataFrame(negative_comments)
    analyze_sentiment_and_extract_kpis(positive_df, negative_df)
    
def main():
    st.set_page_config(
        page_title="Booking.com Reviews",
        
        layout="wide"
    )
    
    st.title("Booking.com Reviews ")
    st.write("Enter a Booking.com hotel URL to extract positive and negative reviews.")

    url = st.text_input("Booking.com URL", placeholder="https://www.booking.com/hotel/...")
    debug_mode = False
    if st.button("Extract Reviews", type="primary"):
        if not url:
            st.error("Please enter a valid Booking.com URL")
        elif not validate_booking_url(url):
            st.error("The URL doesn't appear to be a valid Booking.com hotel URL")
        else:
            with st.spinner("Extracting reviews... This might take a minute or two."):
                # Create a progress section for debugging
                if debug_mode:
                    progress_placeholder = st.empty()
                    progress_placeholder.info("Starting extraction process...")
                    
                    # Override logging functions
                    def log_progress(message):
                        current_content = progress_placeholder.info(message)
                        return current_content
                    
                    # Monkeypatch st.info for debugging
                    orig_info = st.info
                    st.info = lambda msg: log_progress(f"{progress_placeholder.text}\n{msg}" if progress_placeholder.text else msg)
                    
                    reviews = extract_reviews(url, debug_mode)
                    
                    # Restore original function
                    st.info = orig_info
                else:
                    reviews = extract_reviews(url, debug_mode)
                
            if reviews:
                # Create separate dataframes for positive and negative comments
                 create_sentiment_dataframes(reviews)
           
            else:
                st.error("No reviews found or error occurred during extraction")
                if debug_mode and os.path.exists("debug_screenshot.png"):
                    st.image("debug_screenshot.png", caption="Screenshot of the page (for debugging)")

if __name__ == "__main__":
    main()

