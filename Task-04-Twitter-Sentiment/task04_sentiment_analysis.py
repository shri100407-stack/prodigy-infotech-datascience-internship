import pandas as pd
import matplotlib.pyplot as plt
import re
# STEP 1: LOAD THE DATA
# The CSV file does not have column headers, so we give our own names
column_names = ["tweet_id", "entity", "sentiment", "text"]

# Load the dataset from your local disk
# (download twitter_training.csv from the Prodigy GitHub repo and place it here)
file_path = r"D:/internship/task 4/twitter_training.csv"
df = pd.read_csv(file_path, header=None, names=column_names)

# Let's take a first look at the data
print("Number of rows and columns:", df.shape)
print(df.head())

# STEP 2: CLEAN THE DATA
# Some tweets have no text at all. We remove those rows.
df = df.dropna(subset=["text"])

# Some tweets appear more than once in the file. We remove exact duplicates.
df = df.drop_duplicates()

def clean_tweet(tweet):
    tweet = str(tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+", "", tweet)      # remove links
    tweet = re.sub(r"www\S+", "", tweet)       # remove links
    tweet = re.sub(r"@\w+", "", tweet)         # remove @mentions
    tweet = re.sub(r"[^a-z\s]", "", tweet)     # keep only letters and spaces
    tweet = tweet.strip()
    return tweet

# Apply the cleaning function to every tweet in the "text" column
df["clean_text"] = df["text"].apply(clean_tweet)

# Count how many words are in each cleaned tweet (we use this in Chart 4)
def count_words(text):
    words = text.split()
    return len(words)
df["word_count"] = df["clean_text"].apply(count_words)
print("\nNumber of rows after cleaning:", df.shape)
print("\nHow many tweets fall into each sentiment:")
print(df["sentiment"].value_counts())
# STEP 3: MAKE THE CHARTS
# Colours used for the charts (same dark theme as Task 02 and Task 03)
plt.style.use("dark_background")
background_color = "#0d1117"
color_blue  = "#4da6ff"
color_teal  = "#2ec4b6"
color_coral = "#ff6f61"
color_gold  = "#ffd166"
color_white = "#f5f5f5"
my_colors = [color_blue, color_coral, color_teal, color_gold]

# Create one big figure that will hold all 4 charts
fig = plt.figure(figsize=(16, 10), facecolor=background_color)
fig.suptitle("PRODIGY INFOTECH — DATA SCIENCE TASK 04",
             fontsize=22, fontweight="bold", color=color_white, y=0.98)
fig.text(0.5, 0.945, "Sentiment Analysis of Social Media (Twitter) Data",
          fontsize=13, color=color_teal, ha="center")

#CHART 1: Overall sentiment distribution 
# This shows how many tweets are Positive, Negative, Neutral, or Irrelevant
sentiment_counts = df["sentiment"].value_counts()

ax1 = fig.add_subplot(2, 2, 1)
ax1.bar(sentiment_counts.index, sentiment_counts.values, color=my_colors)
ax1.set_title("Overall Sentiment Distribution", color=color_white)
ax1.set_xlabel("Sentiment", color=color_white)
ax1.set_ylabel("Number of Tweets", color=color_white)
ax1.tick_params(colors=color_white)
#CHART 2: Top 10 most talked-about entities/brands 
top_entities = df["entity"].value_counts().head(10)

ax2 = fig.add_subplot(2, 2, 2)
ax2.barh(top_entities.index[::-1], top_entities.values[::-1], color=color_blue)
ax2.set_title("Top 10 Entities by Tweet Volume", color=color_white)
ax2.set_xlabel("Number of Tweets", color=color_white)
ax2.tick_params(colors=color_white)

# CHART 3: Sentiment breakdown for top 6 entities 
# This shows, for the most popular brands, how many tweets were
# positive/negative/neutral/irrelevant
top6_entities = df["entity"].value_counts().head(6).index
df_top6 = df[df["entity"].isin(top6_entities)]

# crosstab makes a table counting sentiment per entity
sentiment_by_entity = pd.crosstab(df_top6["entity"], df_top6["sentiment"])
sentiment_by_entity = sentiment_by_entity.loc[top6_entities]  # keep same order

ax3 = fig.add_subplot(2, 2, 3)
sentiment_by_entity.plot(kind="bar", stacked=True, ax=ax3, color=my_colors)
ax3.set_title("Sentiment Breakdown — Top 6 Entities", color=color_white)
ax3.set_xlabel("Entity", color=color_white)
ax3.set_ylabel("Number of Tweets", color=color_white)
ax3.tick_params(colors=color_white, labelrotation=30)
ax3.legend(facecolor=background_color, edgecolor=color_white, labelcolor=color_white, fontsize=8)

#  CHART 4: Tweet length compared across sentiments
# This shows whether longer or shorter tweets tend to be more positive/negative
ax4 = fig.add_subplot(2, 2, 4)

sentiment_list = df["sentiment"].unique()
for i in range(len(sentiment_list)):
    current_sentiment = sentiment_list[i]
    current_color = my_colors[i % len(my_colors)]
    tweet_lengths = df[df["sentiment"] == current_sentiment]["word_count"]
    ax4.hist(tweet_lengths, bins=30, alpha=0.5, label=current_sentiment, color=current_color)

ax4.set_title("Tweet Length Distribution by Sentiment", color=color_white)
ax4.set_xlabel("Word Count", color=color_white)
ax4.set_ylabel("Frequency", color=color_white)
ax4.set_xlim(0, 60)
ax4.tick_params(colors=color_white)
ax4.legend(facecolor=background_color, edgecolor=color_white, labelcolor=color_white, fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.93])

# STEP 4: SAVE THE CHART
save_path = r"D:/internship/task 4/task04_sentiment_analysis.png"
plt.savefig(save_path, dpi=300, facecolor=background_color, bbox_inches="tight")
print("\nChart saved to:", save_path)

plt.show()
# STEP 5: PRINT A SUMMARY OF WHAT WE FOUND
most_common_sentiment = sentiment_counts.idxmax()
most_common_count = sentiment_counts.max()
most_discussed_entity = top_entities.idxmax()
most_discussed_count = top_entities.max()
print("\n--- KEY FINDINGS ---")
print("Total tweets analysed:", len(df))
print("Most common sentiment:", most_common_sentiment, "(", most_common_count, "tweets )")
print("Most discussed entity:", most_discussed_entity, "(", most_discussed_count, "tweets )")
