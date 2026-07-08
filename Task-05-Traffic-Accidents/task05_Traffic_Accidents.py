# STEP 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# STEP 2: Settings (change these if your setup is different)
FILE_PATH = "D:/internship/task 5/US_Accidents_March23.csv"
OUTPUT_PATH = "D:/internship/task 5/task05_combined_output.png"

SAMPLE_SIZE = 100000     # how many rows we want to work with
CHUNK_SIZE = 200000      # how many rows to read at a time
SAMPLE_FRACTION = 0.08   # % of each chunk we randomly keep

# Dark theme colors (same style used in previous tasks)
BG_COLOR = "#0d1117"
TEXT_COLOR = "#ffffff"
GRID_COLOR = "#30363d"
COLOR_1 = "#58a6ff"   # blue
COLOR_2 = "#f78166"   # orange
COLOR_3 = "#3fb950"   # green
COLOR_4 = "#d29922"   # yellow
COLOR_5 = "#bc8cff"   # purple


# STEP 3: Function to load a random sample from a huge CSV file
def load_sampled_data(filepath, chunk_size, sample_fraction, sample_size):
    print("Loading data in chunks, please wait...")

    sampled_chunks = []
    chunk_number = 0

    # only load the columns we actually need (much faster + less memory)
    columns_needed = [
        "Severity", "Start_Time", "Start_Lat", "Start_Lng",
        "City", "State", "Weather_Condition",
        "Junction", "Traffic_Signal", "Crossing", "Stop",
        "Sunrise_Sunset"
    ]

    for chunk in pd.read_csv(filepath, chunksize=chunk_size,
                              usecols=columns_needed, low_memory=False):
        chunk_number = chunk_number + 1
        small_piece = chunk.sample(frac=sample_fraction, random_state=42)
        sampled_chunks.append(small_piece)
        print("Processed chunk number:", chunk_number)

    combined_data = pd.concat(sampled_chunks, ignore_index=True)

    # if we ended up with more rows than we need, trim it down randomly
    if len(combined_data) > sample_size:
        combined_data = combined_data.sample(n=sample_size, random_state=42)

    print("Final sample size:", len(combined_data), "rows")
    return combined_data


# STEP 4: Function to clean the data
def clean_data(df):
   
    df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng", "Severity"])

    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    df["Hour"] = df["Start_Time"].dt.hour

    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")

    return df


# STEP 5: Function to style each subplot with the dark theme
def style_axis(ax, title):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)

# STEP 6: Plot 1 - Top weather conditions during accidents
def plot_weather(ax, df):
    top_weather = df["Weather_Condition"].value_counts().head(8)

    ax.barh(top_weather.index[::-1], top_weather.values[::-1], color=COLOR_1)
    style_axis(ax, "Top Weather Conditions During Accidents")
    ax.set_xlabel("Number of Accidents")


# STEP 7: Plot 2 - Accidents by hour of day
def plot_time_of_day(ax, df):
    hourly_counts = df.groupby("Hour").size()

    ax.plot(hourly_counts.index, hourly_counts.values,
            color=COLOR_2, marker="o", linewidth=2)
    ax.fill_between(hourly_counts.index, hourly_counts.values,
                     color=COLOR_2, alpha=0.2)

    style_axis(ax, "Accidents by Hour of Day")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Number of Accidents")


# STEP 8: Plot 3 - Road-related contributing factors
def plot_road_features(ax, df):
    road_features = ["Junction", "Traffic_Signal", "Crossing", "Stop"]
    feature_counts = []

    for feature in road_features:
        count_true = df[feature].sum()
        feature_counts.append(count_true)

    ax.bar(road_features, feature_counts,
           color=[COLOR_1, COLOR_2, COLOR_3, COLOR_4])
    style_axis(ax, "Accidents Near Road Features")
    ax.set_ylabel("Number of Accidents")


# STEP 9: Plot 4 - Accident hotspots (location scatter)
def plot_hotspots(ax, df):
    scatter = ax.scatter(df["Start_Lng"], df["Start_Lat"],
                          c=df["Severity"], cmap="plasma",
                          s=5, alpha=0.5)

    style_axis(ax, "Accident Hotspots (colored by Severity)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.set_label("Severity", color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=TEXT_COLOR)

# STEP 10: Build the combined 2x2 dashboard figure
def build_combined_figure(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG_COLOR)

    fig.suptitle("PRODIGY INFOTECH \u2014 DATA SCIENCE TASK 05",
                 color=TEXT_COLOR, fontsize=20, fontweight="bold", y=0.98)
    fig.text(0.5, 0.945,
             "Traffic Accident Analysis: Weather, Time & Road Conditions",
             color="#8b949e", fontsize=12, ha="center")

    plot_weather(axes[0, 0], df)
    plot_time_of_day(axes[0, 1], df)
    plot_road_features(axes[1, 0], df)
    plot_hotspots(axes[1, 1], df)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    print("Saved combined dashboard to:", output_path)
    plt.show()

# STEP 11: Main function - runs everything in order
def main():
    raw_data = load_sampled_data(FILE_PATH, CHUNK_SIZE,
                                  SAMPLE_FRACTION, SAMPLE_SIZE)
    clean = clean_data(raw_data)
    build_combined_figure(clean, OUTPUT_PATH)

if __name__ == "__main__":
    main()
