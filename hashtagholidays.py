import json
import os
import random
import time
from datetime import datetime
from typing import Optional

import instaloader
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from transformers import pipeline

app = FastAPI()
sessions = {}
tagger = SequenceTagger.load("ner")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    tokenizer="facebook/bart-large-mnli",
)
all_data_global = None
sorted_place_scores_global = None
RESULTS_PER_PAGE = 5


def get_session():
    if "session" not in sessions:
        sessions["session"] = {}
    return sessions["session"]


class SetOrigin(BaseModel):
    origin: str


@app.post("/set-origin/")
def set_origin(data: SetOrigin, session: dict = Depends(get_session)):
    session["origin"] = data.origin
    session["current_page"] = 1
    return {"message": f"Origin set to {data.origin}."}


class SetDistance(BaseModel):
    distance: int


@app.post("/set-distance/")
def set_distance(data: SetDistance, session: dict = Depends(get_session)):
    session["distance_threshold"] = data.distance
    session["current_page"] = 1
    return {"message": f"Distance threshold set to {data.distance} miles."}


@app.get("/get-top-recommendations/")
def get_top_recommendations(
    sort_by: Optional[str] = "score",
    page: Optional[int] = 1,
    session: dict = Depends(get_session),
):
    user_origin = session.get("origin", "San Jose")
    user_distance_threshold = session.get("distance_threshold", 500)
    recommendations = get_recommendations(
        user_location=user_origin,
        distance_threshold=user_distance_threshold,
        sort_by=sort_by,
        session=session,
    )
    start_index = (page - 1) * RESULTS_PER_PAGE
    end_index = start_index + RESULTS_PER_PAGE
    paginated_recommendations = dict(
        list(recommendations.items())[start_index:end_index]
    )
    return {"recommendations": paginated_recommendations}


@app.get("/places")
def get_all_places():
    if all_places := list(places.find({})):
        return {"places": [place["name"] for place in all_places]}
    else:
        raise HTTPException(status_code=404, detail="No places found in the database.")


@app.get("/places/scores")
def get_all_place_scores():
    if scores := get_sorted_place_scores():
        return scores
    else:
        raise HTTPException(status_code=404, detail="No place scores found.")


@app.post("/reset-session/")
def reset_session(session: dict = Depends(get_session)):
    session.clear()
    return {"message": "Session reset successfully."}


@app.post("/dialogflow-webhook/")
def handle_dialogflow_request(request: dict):
    session = get_session()

    intent_name = request["queryResult"]["intent"]["displayName"]

    if intent_name == "set-origin":
        city = request["queryResult"]["parameters"].get("city", "")
        state = request["queryResult"]["parameters"].get("state", "")
        county_data = request["queryResult"]["parameters"].get("county", {})

        if isinstance(county_data, dict):
            admin_area = county_data.get("admin-area", "")
            subadmin_area = county_data.get("subadmin-area", "")
            county_city = county_data.get("city", "")
            street_address = county_data.get("street-address", "")
        else:
            admin_area = ""
            subadmin_area = ""
            county_city = ""
            street_address = ""
            city = county_data

        if not city:
            city = request["queryResult"]["parameters"].get("city", "")

        state = request["queryResult"]["parameters"].get("state", "")

        origin_parts = [
            part
            for part in [
                street_address,
                subadmin_area,
                admin_area,
                county_city,
                city,
                state,
            ]
            if part
        ]
        origin = ", ".join(origin_parts)

        if not origin:
            return {"fulfillmentText": "Please specify the place name for the origin."}
        response = set_origin(SetOrigin(origin=origin), session)
        if "distance_threshold" in session:
            return {
                "fulfillmentText": f"📍 Origin set to {session['origin']}.\n🚗 You can now ask for place recommendations."
            }
        else:
            return {
                "fulfillmentText": f"📍 Origin set to {session['origin']}.\n🔢 Please specify the distance threshold next."
            }

    elif intent_name == "set-distance":
        distance = request["queryResult"]["parameters"].get("distance")
        if not distance:
            return {"fulfillmentText": "Please specify a valid distance value."}
        response = set_distance(SetDistance(distance=distance), session)
        if "origin" in session:
            return {
                "fulfillmentText": f"🔢 Distance threshold set to {session['distance_threshold']} miles.\n🚗 You can now ask for place recommendations."
            }
        else:
            return {
                "fulfillmentText": f"🔢 Distance threshold set to {session['distance_threshold']} miles.\n📍 Please specify the place name for the origin next."
            }

    elif intent_name == "get-recommendations":
        if "origin" not in session and "distance_threshold" not in session:
            return {
                "fulfillmentText": "Please specify the 🌍 place name for the origin and 🔢 distance threshold before asking for recommendations."
            }
        if "origin" not in session:
            return {
                "fulfillmentText": "Please specify the 🌍 place name for the origin before asking for recommendations."
            }
        elif "distance_threshold" not in session:
            return {
                "fulfillmentText": "Please specify the 🔢 distance threshold before asking for recommendations."
            }
        session.setdefault("current_page", 1)
        response = get_top_recommendations(
            page=session["current_page"], session=session
        )
        recommendations = response.get("recommendations", {})

        if not recommendations:
            session["current_page"] = 1
            return {
                "fulfillmentText": "🙁 Sorry, I couldn't find any recommendations based on your preferences."
            }

        recommendation_texts = []
        for place, details in recommendations.items():
            distance = details.get("distance", "unknown distance")
            duration = details.get("duration", "unknown duration")
            recommendation_texts.append(
                # f"📍 {place} \n      - Distance: {distance:.2f} miles\n      - Duration: {duration}"
                f"📍 {place} \n      - 🛣️ {distance:.2f} miles\n      - ⏰ {duration} away"
            )

        formatted_recommendations = "\n\n".join(recommendation_texts)
        return {
            "fulfillmentText": f"Here are some top recommendations for you: 🎉 \n\n Origin is {session['origin']}.\n\n Destinations:\n{formatted_recommendations} \n\nFor more recommendations, simply say 'next' or 'give me some more', and I'll be happy to provide additional options for you."
        }
    elif intent_name == "next-recommendations":
        if "origin" not in session and "distance_threshold" not in session:
            return {
                "fulfillmentText": "Please specify the 🌍 place name for the origin and 🔢 distance threshold before asking for recommendations."
            }
        if "origin" not in session:
            return {
                "fulfillmentText": "Please specify the 🌍 place name for the origin before asking for recommendations."
            }
        elif "distance_threshold" not in session:
            return {
                "fulfillmentText": "Please specify the 🔢 distance threshold before asking for recommendations."
            }
        session["current_page"] = session.get("current_page", 1) + 1
        response = get_top_recommendations(
            page=session["current_page"], session=session
        )
        recommendations = response.get("recommendations", {})

        if not recommendations:
            return {"fulfillmentText": "🙁 Sorry, there are no more recommendations."}

        recommendation_texts = []
        for place, details in recommendations.items():
            distance = details.get("distance", "unknown distance")
            duration = details.get("duration", "unknown duration")
            recommendation_texts.append(
                f"📍 {place} \n      - 🛣️ {distance:.2f} miles\n      - ⏰ {duration} away"
            )

        formatted_recommendations = "\n\n".join(recommendation_texts)
        return {
            "fulfillmentText": f"Here are more recommendations for you: 🎉 \n\n Origin is {session['origin']}.\n\n Destinations:\n{formatted_recommendations}"
        }

    elif intent_name == "reset-session":
        reset_session(session)
        return {"fulfillmentText": "🔄 Session has been reset successfully."}

    else:
        return {"fulfillmentText": "🤔 Sorry, I couldn't understand that."}


# Load environment variables
load_dotenv("/Users/rajsudharshan/work/Hushh/.env")

# MongoDB setup
uri = f"mongodb+srv://{os.getenv('MONGO_DB_USERNAME')}:{os.getenv('MONGO_DB_PASSWORD')}@hashtagholidays.himfbcs.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi("1"))
db = client["HashtagHolidaysDB"]

instagram_posts = db["instagram_posts"]
places = db["places"]


# Load Instagram data
def extract_urls_and_timestamps_from_files(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
        saved_data = [
            (
                entry["string_map_data"]["Saved on"]["href"],
                entry["string_map_data"]["Saved on"]["timestamp"],
            )
            for entry in data.get("saved_saved_media", [])
        ]
        liked_data = [
            (
                entry["string_list_data"][0]["href"],
                entry["string_list_data"][0]["timestamp"],
            )
            for entry in data.get("likes_media_likes", [])
        ]
    return saved_data + liked_data


def get_all_data():
    global all_data_global
    if all_data_global is None:
        all_data_global = extract_urls_and_timestamps_from_files(
            os.getenv("SAVED_POSTS_PATH")
        )
        all_data_global += extract_urls_and_timestamps_from_files(
            os.getenv("LIKED_POSTS_PATH")
        )
    return all_data_global


L = instaloader.Instaloader()


def set_random_proxy():
    proxy = random.choice(os.getenv("PROXIES").split(","))
    proxies = {"http": f"http://{proxy}", "https": f"https://{proxy}"}
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.proxies = proxies
    L.context._session = session


failed_urls = []


def fetch_post_details(url):
    shortcode = url.split("/")[-2]

    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return post.caption, ", ".join(post.caption_hashtags)
    except instaloader.exceptions.BadResponseException:
        print(f"Failed to fetch metadata for URL: {url}")
        failed_urls.append(url)
        instagram_posts.insert_one({"url": url})
        return None, None


def classify_caption(caption):
    return classifier(caption, ["travel", "food", "other"])["labels"][0]


def post_exists(url):
    return instagram_posts.find_one({"url": url}) is not None


def unix_to_datetime(unix_timestamp):
    return datetime.utcfromtimestamp(unix_timestamp)


def store_posts(url, caption, hashtags, category, timestamp):
    if category in ["travel", "food"]:
        return instagram_posts.insert_one(
            {
                "url": url,
                "caption": caption,
                "hashtags": hashtags,
                "category": category,
                "timestamp": unix_to_datetime(timestamp),
            }
        ).inserted_id
    else:
        return instagram_posts.insert_one(
            {"url": url, "category": category}
        ).inserted_id


def store_place_in_db(place, timestamp, post_id):
    if not places.find_one({"name": place, "post_id": post_id}):
        places.insert_one(
            {
                "name": place,
                "timestamp": unix_to_datetime(timestamp),
                "post_id": post_id,
            }
        )


def remove_hashtags(text):
    if text:
        return " ".join([word for word in text.split() if not word.startswith("#")])


def decay_score(days_elapsed, lambda_value=0.1):
    return 2 ** (-lambda_value * days_elapsed)


def get_top_places():
    current_date = datetime.utcnow()
    all_places = places.find({})
    place_scores = {}

    for place_entry in all_places:
        days_elapsed = (current_date - place_entry["timestamp"]).days
        score = decay_score(days_elapsed)
        if place_entry["name"] in place_scores:
            place_scores[place_entry["name"]] += score
        else:
            place_scores[place_entry["name"]] = score

    return dict(sorted(place_scores.items(), key=lambda item: item[1], reverse=True))


def extract_visited_places_from_location_history(
    file_path=os.getenv("LOCATION_HISTORY_PATH"),
):
    with open(file_path, "r") as file:
        data = json.load(file)

    visited_places = [
        place_visit["location"]["address"]
        for timeline_object in data["timelineObjects"]
        if "placeVisit" in timeline_object
        for place_visit in [timeline_object["placeVisit"]]
    ]
    visited_places = [
        "Irvine",
    ]
    return visited_places


def has_been_visited(place, visited_places):
    return any(visited_place in place for visited_place in visited_places)


GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_distances_from_google_maps(user_location, destinations):
    endpoint = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters,status,condition",
    }
    origins = [user_location]

    # Function to split destinations into chunks of 50
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    results = {}
    for chunked_destinations in chunks(destinations, 49):
        data = {
            "origins": [{"waypoint": {"address": origin}} for origin in origins],
            "destinations": [
                {"waypoint": {"address": destination}}
                for destination in chunked_destinations
            ],
            "travelMode": "DRIVE",
        }
        response = requests.post(endpoint, headers=headers, json=data)
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            print(
                f"Failed to decode JSON for user location {user_location} and destinations {chunked_destinations}. Response text: {response.text}"
            )
            continue

        try:
            for result in response_data:
                if (
                    "condition" in result.keys()
                    and result["condition"] == "ROUTE_EXISTS"
                ):
                    destination_index = result["destinationIndex"]
                    destination_name = chunked_destinations[destination_index]
                    distance_miles = result.get("distanceMeters", 0) / 1609.34
                    duration_seconds = int(
                        result.get("duration", "0s").replace("s", "")
                    )
                    hours = duration_seconds // 3600
                    minutes = (duration_seconds % 3600) // 60
                    duration_str = f"{hours}h {minutes}m"
                    results[destination_name] = {
                        "distance": distance_miles,
                        "duration": duration_str,
                    }
        except (KeyError, IndexError):
            continue

    return results


def get_sorted_place_scores():
    global sorted_place_scores_global
    if sorted_place_scores_global is None:
        set_random_proxy()
        sorted_place_scores_global = get_top_places()
    return sorted_place_scores_global


test_urls = [
    ("https://www.instagram.com/p/CvunpwdrA8P/", 1691614394),
    ("https://www.instagram.com/reel/CvgSrl9Ak6Q/", 1692223171),
    ("https://www.instagram.com/reel/CvfnUJHLj58/", 1692223171),
    ("https://www.instagram.com/p/Cv5IM6Yhx3E/", 1692223171),
    ("https://www.instagram.com/p/CwBJCtFgZL-/", 1692223171),
    ("https://www.instagram.com/reel/CvsgKpdNt2A/", 1692223171),
    ("https://www.instagram.com/reel/Cu281ZKJDcp/", 1692161969),
    ("https://www.instagram.com/reel/CvJB6BApC2e/", 1692161969),
    ("https://www.instagram.com/p/CvqywNhgjaR/", 1692223171),
    ("https://www.instagram.com/p/CueF4UZybRr/", 1691614394),
    ("https://www.instagram.com/p/Cd8Ui3cjT6U/", 1692163385),
    ("https://www.instagram.com/reel/CcYBfB9l6fm/", 1691625794),
    ("https://www.instagram.com/p/Cvck00-tg7s/", 1691637194),
    ("https://www.instagram.com/p/CuCcCtsPdIy/", 1691648594),
    ("https://www.instagram.com/p/Ct_3pZmuyqV/", 1691659994),
    ("https://www.instagram.com/reel/CvEfz6KhyW9/", 1691671394),
    ("https://www.instagram.com/p/CvpkVSnN-Pz/", 1691682794),
    ("https://www.instagram.com/p/CvP-aIihE7K/", 1691694194),
    ("https://www.instagram.com/p/ClGZaOEggTN/", 1691705594),
    ("https://www.instagram.com/p/Cv0wzKRN7sf/", 1691716994),
]


def get_recommendations(
    user_location, session: dict, distance_threshold=1000, sort_by="score"
):
    sorted_place_scores = session.get("sorted_place_scores")
    if not sorted_place_scores:
        update_place_scores_from_history_and_posts(session)
        sorted_place_scores = session["sorted_place_scores"]
    destinations = list(sorted_place_scores.keys())
    results = get_distances_from_google_maps(user_location, destinations)

    visited_places = extract_visited_places_from_location_history()

    recommendations = {}
    for place, data in results.items():
        distance = data.get("distance")
        if distance is not None and distance <= distance_threshold:
            if has_been_visited(place, visited_places):
                sorted_place_scores[place] *= 0.8
            sorted_place_scores[place] *= 1.5
            recommendations[place] = {
                "score": sorted_place_scores[place],
                "distance": data.get("distance"),
                "duration": data.get("duration"),
            }

    if sort_by == "distance":
        return dict(
            sorted(
                recommendations.items(),
                key=lambda item: (item[1]["distance"], -item[1]["score"]),
            )
        )
    elif sort_by == "time":

        def duration_to_minutes(duration_str):
            hours, minutes = map(int, duration_str.replace("m", "").split("h"))
            return hours * 60 + minutes

        return dict(
            sorted(
                recommendations.items(),
                key=lambda item: (
                    duration_to_minutes(item[1]["duration"]),
                    -item[1]["score"],
                ),
            )
        )
    else:
        return dict(
            sorted(
                recommendations.items(),
                key=lambda item: (-item[1]["score"], item[1]["distance"]),
            )
        )


def update_place_scores_from_history_and_posts(session: dict):
    all_data = get_all_data()
    for url, timestamp in test_urls:
        if not post_exists(url):
            time.sleep(random.uniform(1, 3))

            caption, hashtags = fetch_post_details(url)
            processed_caption = remove_hashtags(caption)
            if processed_caption and len(processed_caption) > 3:
                category = classify_caption(processed_caption)
                processed_text = processed_caption + "\n" + hashtags
                post_id = store_posts(url, caption, hashtags, category, timestamp)
                sentence = Sentence(processed_text)
                tagger.predict(sentence)
                for entity in sentence.get_spans("ner"):
                    if "LOC" in entity.tag:
                        store_place_in_db(entity.text, timestamp, post_id)

    sorted_place_scores = get_sorted_place_scores()

    with open(os.getenv("BROWSER_HISTORY_PATH"), "r") as file:
        data = json.load(file)

    titles = [entry["title"] for entry in data["Browser History"]]

    place_counts = {}
    for place in sorted_place_scores.keys():
        count = sum([1 for title in titles if place.lower() in title.lower()])
        place_counts[place] = count

    for place, count in place_counts.items():
        if place in ["San Diego", "Los Angeles", "Irvine", "Hawaii", "newyork"]:
            count += 14500
        sorted_place_scores[place] *= 1 + (0.1 * count)
    sorted_place_scores_dict = dict(
        sorted(sorted_place_scores.items(), key=lambda item: item[1], reverse=True)
    )
    session["sorted_place_scores"] = sorted_place_scores_dict
    return sorted_place_scores_dict
