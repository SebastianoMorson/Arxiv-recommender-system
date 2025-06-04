import frontend
import recommender

if __name__ == "__main__":
    try:
        frontend.start()
        recommender.start_recommender()
    except Exception as e:
        print(f"An error occurred: {e}")