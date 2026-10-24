
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

st.set_page_config(
    page_title="Phone Usage Analysis",
    page_icon="📱",
    layout="wide"
)

# Load the dataset
df = pd.read_csv("phone_usage_india.csv")
engineered_df = pd.read_csv("phone_usage_feature_engineered.csv")

classifier = joblib.load("random_forest_primary_use.pkl")
label_encoder = joblib.load("label_encoder.pkl")

kmeans_model = joblib.load("best_kmeans_model.pkl")
kmeans_scaler = joblib.load("kmeans_scaler.pkl")

# Dashboard title
st.title("📱 Decoding Phone Usage Patterns in India")

# Create dashboard tabs
st.sidebar.title("📱 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Welcome",
        "📊 EDA Visuals",
        "🧠 Predict Primary Usage",
        "👥 User Clustering & Segmentation",
        "📘 Project Info"
    ]
)

if page == "🏠 Welcome":

    #st.title("📱 Decoding Phone Usage Patterns in India")
    st.subheader("Understanding Smartphone Habits Through Data Science")

    st.write(
        "Welcome to an interactive exploration of smartphone usage "
        "patterns in India. This project combines data analysis, "
        "machine learning, and user segmentation to uncover insights "
        "into how people use their phones."
    )

    #st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", f"{len(df):,} users")

    with col2:
        st.metric("Usage Categories", df["Primary Use"].nunique())

    with col3:
        st.metric("Analysis Approach", "Machine Learning")

    st.divider()

    st.subheader("Explore the Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Exploratory Data Analysis")
        st.write(
            "Explore usage distributions, screen time, social media "
            "habits, and relationships between numerical features."
        )

        st.markdown("### 🧠 Predict Primary Usage")
        st.write(
            "Enter a user's phone usage details and use the trained "
            "Random Forest model to predict their likely primary-use category."
        )

    with col2:
        st.markdown("### 👥 User Clustering")
        st.write(
            "Explore groups of users identified using KMeans clustering "
            "based on their phone usage behaviour."
        )

        st.markdown("### 🎯 Project Objective")
        st.write(
            "Turn smartphone usage data into understandable insights "
            "using exploratory analysis and machine learning."
        )

    st.info(
        "Choose a section from the navigation menu on the left to begin."
    )


elif page == "📊 EDA Visuals":
    st.header("Exploratory Data Analysis")

   # st.title("📊 Exploratory Data Analysis")
    st.write(
        "Explore smartphone usage patterns across 17,686 users in India. "
        "These visualisations examine primary usage categories, screen time, "
        "social media habits, and relationships between numerical features."
    )

    st.divider()

    # Display key metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Users", f"{len(df):,}")

    with col2:
        st.metric(
            "Average Screen Time",
            f"{df['Screen Time (hrs/day)'].mean():.2f} hrs/day"
        )

    with col3:
        st.metric(
            "Average Data Usage",
            f"{df['Data Usage (GB/month)'].mean():.2f} GB/month"
        )

    st.divider()

    
    # Charts 1 and 2: Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Distribution of Primary Use")

        primary_counts = df["Primary Use"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        primary_counts.plot(kind="bar", ax=ax)

        ax.set_xlabel("Primary Use")
        ax.set_ylabel("Number of Users")
        ax.set_title("Number of Users by Primary Use")
        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("2. Screen Time Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=df,
            x="Screen Time (hrs/day)",
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_title("Distribution of Daily Screen Time")
        ax.set_xlabel("Screen Time (hours/day)")
        ax.set_ylabel("Number of Users")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    
    # Charts 3 and 4: Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("3. Average Screen Time by Primary Use")

        avg_screen_time = (
            df.groupby("Primary Use")["Screen Time (hrs/day)"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        avg_screen_time.plot(kind="bar", ax=ax)

        ax.set_xlabel("Primary Use")
        ax.set_ylabel("Average Screen Time (hours/day)")
        ax.set_title("Average Screen Time by Primary Use")
        ax.tick_params(axis="x", rotation=30)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("📱 Social Media Time by Primary Use")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(
            data=df,
            x="Social Media Time (hrs/day)",
            hue="Primary Use",
            kde=True,
            ax=ax
        )

        ax.set_title("Social Media Time Distribution by Primary Use")
        ax.set_xlabel("Social Media Time (hours/day)")
        ax.set_ylabel("Number of Users")

        plt.tight_layout()
        st.pyplot(fig)
        #plt.close(fig)


   
    # Charts 5 and 6: Display side by side
   
    # Charts 5 and 6: Display side by side
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📉 Correlation Heatmap")

        # Use the 10 numerical features from the clustering analysis
        numerical_cols = df[
            [
                "Age",
                "Screen Time (hrs/day)",
                "Data Usage (GB/month)",
                "Calls Duration (mins/day)",
                "Number of Apps Installed",
                "Social Media Time (hrs/day)",
                "E-commerce Spend (INR/month)",
                "Streaming Time (hrs/day)",
                "Gaming Time (hrs/day)",
                "Monthly Recharge Cost (INR)"
            ]
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            numerical_cols.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.75},
            ax=ax
        )

        ax.set_title("Correlation Between Numerical Features", fontsize=10)
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.subheader("📦 Screen Time by Primary Use")

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.boxplot(
            data=df,
            x="Primary Use",
            y="Screen Time (hrs/day)",
            ax=ax
        )

        ax.set_title("Screen Time Across Usage Categories", fontsize=10)
        ax.set_xlabel("Primary Use")
        ax.set_ylabel("Screen Time (hours/day)")
        ax.tick_params(axis="x", rotation=35, labelsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)





elif page == "🧠 Predict Primary Usage":
    st.write(
        "Use the trained Random Forest model to estimate a user's "
        "primary phone usage category based on their demographic "
        "details and smartphone usage patterns."
    )

    st.info(
        "Adjust the input values below, then click "
        "'Predict Primary Use' to generate a prediction."
    )

    #st.divider()
    #st.header("🧠 Predict the Primary Use")
    #st.write("Enter the user's details and phone usage patterns.")

    with st.form("user_input_form"):
        input_data = {}

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 10, 100, 25)
            gender = st.selectbox("Gender", sorted(engineered_df["Gender"].dropna().unique()))
            location = st.selectbox("Location", sorted(engineered_df["Location"].dropna().unique()))
            phone_brand = st.selectbox("Phone Brand", sorted(engineered_df["Phone Brand"].dropna().unique()))
            os_value = st.selectbox("Operating System", sorted(engineered_df["OS"].dropna().unique()))

            input_data["Age"] = age
            input_data["Gender"] = gender
            input_data["Location"] = location
            input_data["Phone Brand"] = phone_brand
            input_data["OS"] = os_value

            input_data["E-commerce Spend (INR/month)"] = st.slider(
                "E-commerce Spend (INR/month)", 0, 10000, 500
            )
            input_data["Monthly Recharge Cost (INR)"] = st.slider(
                "Monthly Recharge Cost (INR)", 0, 1000, 200
            )
            

        with col2:
            input_data["Number of Apps Installed"] = st.slider(
                        "Number of Apps Installed", 0, 200, 30
            )
            input_data["Calls Duration (mins/day)"] = st.slider(
                            "Calls Duration (mins/day)", 0, 300, 30
            )
            input_data["Data Usage (GB/month)"] = st.slider(
                "Data Usage (GB/month)", 0, 100, 20
            )
            input_data["Screen Time (hrs/day)"] = st.slider(
                "Screen Time (hrs/day)", 0.0, 24.0, 5.0
            )
            input_data["Social Media Time (hrs/day)"] = st.slider(
                "Social Media Time (hrs/day)", 0.0, 24.0, 2.0
            )
            input_data["Streaming Time (hrs/day)"] = st.slider(
                "Streaming Time (hrs/day)", 0.0, 24.0, 1.0
            )
            input_data["Gaming Time (hrs/day)"] = st.slider(
                "Gaming Time (hrs/day)", 0.0, 24.0, 1.0
            )

        submitted = st.form_submit_button("Predict Primary Use")

    if submitted:
        input_data["Entertainment_Index"] = (
            input_data["Streaming Time (hrs/day)"]
            + input_data["Gaming Time (hrs/day)"]
            + input_data["Social Media Time (hrs/day)"]
        ) / 3

        input_data["Spend_Ratio"] = (
            input_data["E-commerce Spend (INR/month)"]
            / (input_data["Monthly Recharge Cost (INR)"] + 1)
        )

        input_df = pd.DataFrame([input_data])

        prediction = classifier.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)

        st.success(f"✅ Predicted Primary Use: **{predicted_label[0]}**")






elif page == "👥 User Clustering & Segmentation":

    st.subheader("👥 User Clustering & Segmentation")

    st.write(
        "Explore smartphone user groups based on their usage "
        "patterns using KMeans clustering."
    )

    # --------------------------------------------------
    # 1. PREPARE DATA AND ASSIGN CLUSTERS
    # --------------------------------------------------

    clustering_features = [
        "Age",
        "Screen Time (hrs/day)",
        "Data Usage (GB/month)",
        "Calls Duration (mins/day)",
        "Number of Apps Installed",
        "Social Media Time (hrs/day)",
        "E-commerce Spend (INR/month)",
        "Streaming Time (hrs/day)",
        "Gaming Time (hrs/day)",
        "Monthly Recharge Cost (INR)"
    ]

    X_cluster = df[clustering_features].copy()

    X_cluster_scaled = kmeans_scaler.transform(X_cluster)

    df_clustered = df.copy()

    df_clustered["Cluster"] = kmeans_model.predict(
        X_cluster_scaled
    )

    cluster_counts = (
        df_clustered["Cluster"]
        .value_counts()
        .sort_index()
    )

    cluster_summary = (
        df_clustered.groupby("Cluster")[clustering_features]
        .mean()
        .round(2)
    )

    # --------------------------------------------------
    # 2. COMPACT CLUSTER OVERVIEW
    # --------------------------------------------------

    st.subheader("📊 Cluster Overview")

    overview_cols = st.columns(2 + len(cluster_counts))

    with overview_cols[0]:
        st.metric(
            "Clusters",
            df_clustered["Cluster"].nunique()
        )

    with overview_cols[1]:
        st.metric(
            "Total Users",
            f"{len(df_clustered):,}"
        )

    for i, (cluster_id, count) in enumerate(
        cluster_counts.items()
    ):
        with overview_cols[i + 2]:
            st.metric(
                f"Cluster {cluster_id}",
                f"{count:,}"
            )


    # --------------------------------------------------
    # 3. COMPARISON TABLE AND CHART SIDE BY SIDE
    # --------------------------------------------------

   
    col_table, col_chart = st.columns([1.5, 1])

    with col_table:
        st.subheader("📋 Compare Usage Patterns")

        st.dataframe(
            cluster_summary.T,
            use_container_width=True,
            height=320
        )

    with col_chart:
        st.subheader("💰 E-commerce Spending")

        spending = cluster_summary[
            "E-commerce Spend (INR/month)"
        ]

        fig, ax = plt.subplots(figsize=(4, 3.5))

        spending.plot(
            kind="bar",
            ax=ax,
            width=0.55
        )

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Average Spend (₹)")
        ax.tick_params(axis="x", rotation=0)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="₹%.0f",
                padding=2,
                fontsize=9
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # --------------------------------------------------
    # 4. KEY FINDING
    # --------------------------------------------------

    lower_cluster = spending.idxmin()
    higher_cluster = spending.idxmax()

    if lower_cluster == higher_cluster:
        st.caption(
            "All clusters have the same average e-commerce spending."
        )
    else:
        st.info(
            f"**Key finding:** Cluster {lower_cluster} has the "
            f"lower average monthly e-commerce spend "
            f"(₹{spending[lower_cluster]:,.0f}), while Cluster "
            f"{higher_cluster} has the higher average "
            f"(₹{spending[higher_cluster]:,.0f})."
        )

    st.caption(
        "Clusters represent patterns in the dataset; they do not "
        "establish users' motivations."
    )

    # --------------------------------------------------
    # 5. INTERACTIVE USER CLUSTER PREDICTION
    # --------------------------------------------------

    with st.expander("📱 Try a User Cluster Prediction"):

        st.write(
            "Adjust the sliders to enter a user's smartphone "
            "usage details and predict their cluster."
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=70,
                value=25,
                key="cluster_age"
            )

            screen_time = st.slider(
                "Screen Time (hrs/day)",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5,
                key="cluster_screen_time"
            )

            data_usage = st.slider(
                "Data Usage (GB/month)",
                min_value=0.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                key="cluster_data_usage"
            )

            calls = st.slider(
                "Calls Duration (mins/day)",
                min_value=0,
                max_value=300,
                value=60,
                key="cluster_calls"
            )

            apps = st.slider(
                "Number of Apps Installed",
                min_value=0,
                max_value=200,
                value=30,
                key="cluster_apps"
            )

        with col2:
            social_media = st.slider(
                "Social Media Time (hrs/day)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                key="cluster_social"
            )

            ecommerce = st.slider(
                "E-commerce Spend (INR/month)",
                min_value=0,
                max_value=20000,
                value=2500,
                step=500,
                key="cluster_ecommerce"
            )

            streaming = st.slider(
                "Streaming Time (hrs/day)",
                min_value=0.0,
                max_value=10.0,
                value=1.5,
                step=0.5,
                key="cluster_streaming"
            )

            gaming = st.slider(
                "Gaming Time (hrs/day)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                key="cluster_gaming"
            )

            recharge = st.slider(
                "Monthly Recharge Cost (INR)",
                min_value=0,
                max_value=1000,
                value=200,
                step=50,
                key="cluster_recharge"
            )

        # --------------------------------------------------
        # 6. CREATE INPUT DATAFRAME
        # --------------------------------------------------

        user_data = {
            "Age": age,
            "Screen Time (hrs/day)": screen_time,
            "Data Usage (GB/month)": data_usage,
            "Calls Duration (mins/day)": calls,
            "Number of Apps Installed": apps,
            "Social Media Time (hrs/day)": social_media,
            "E-commerce Spend (INR/month)": ecommerce,
            "Streaming Time (hrs/day)": streaming,
            "Gaming Time (hrs/day)": gaming,
            "Monthly Recharge Cost (INR)": recharge
        }

        user_df = pd.DataFrame(
            [user_data],
            columns=clustering_features
        )

        # --------------------------------------------------
        # 7. PREDICT USER CLUSTER
        # --------------------------------------------------

        user_scaled = kmeans_scaler.transform(user_df)

        predicted_cluster = kmeans_model.predict(
            user_scaled
        )[0]

        st.success(
            f"Predicted segment: Cluster {predicted_cluster}"
        )

        # --------------------------------------------------
        # 8. DISPLAY ENTERED DETAILS
        # --------------------------------------------------

        st.write("**Your Entered Details**")

        st.dataframe(
            user_df.T.rename(columns={0: "Value"}),
            use_container_width=True
        )

elif page == "📘 Project Info":

    st.subheader("📘 Project Information")

    st.write(
        "An overview of the objectives, dataset, methods, "
        "tools, findings, and limitations of this project."
    )

    # 1. Project objective
    st.markdown("### 🎯 Project Objective")

    st.write(
        "This project explores smartphone usage patterns in India "
        "using data analysis and machine learning. It aims to "
        "understand how users interact with their phones, examine "
        "relationships between usage features, predict primary "
        "phone usage categories, and identify groups of users "
        "with similar usage patterns."
    )

    # 2. Dataset
    st.markdown("### 📂 Dataset Overview")

    st.write(
        f"The dataset contains **{len(df):,} user records** "
        f"and {df.shape[1]} columns in the original dataset."
    )

    st.write("The dataset includes information about:")

    st.markdown(
        """
        - **Demographics:** Age and gender
        - **Device information:** Phone brand and operating system
        - **Phone usage:** Screen time, calls, data usage and installed apps
        - **Digital activities:** Social media, streaming and gaming
        - **Spending:** E-commerce expenditure and monthly recharge cost
        - **Target category:** Primary Use
        """
    )

    # 3. Tools
    st.markdown("### 🛠️ Tools and Technologies")

    st.markdown(
        """
        - **Python** – Main programming language
        - **Pandas** – Data loading, manipulation and analysis
        - **Matplotlib and Seaborn** – Data visualisation
        - **Scikit-learn** – Machine learning and clustering
        - **Joblib** – Saving and loading trained models
        - **Streamlit** – Interactive web dashboard
        """
    )

    # 4. Methodology
    st.markdown("### ⚙️ Methodology")

    st.markdown(
        """
        **1. Exploratory Data Analysis (EDA)**

        Examined primary-use categories, screen-time distributions,
        usage patterns and correlations between numerical features.

        **2. Feature Engineering**

        Created `Entertainment_Index` from streaming, gaming and
        social media time, and `Spend_Ratio` from e-commerce spending
        and monthly recharge cost.

        **3. Primary Use Classification**

        Used a trained Random Forest classifier to predict a user's
        primary phone usage category from their input features.

        **4. User Clustering**

        Applied KMeans clustering to group users based on ten
        numerical demographic and smartphone usage features.

        **5. Interactive Dashboard**

        Developed a Streamlit application to explore the analysis,
        obtain predictions and inspect cluster profiles.
        """
    )

    # 5. Key findings
    st.markdown("### 📊 Key Findings")

    primary_counts = df["Primary Use"].value_counts()

    st.write(
        f"The dataset contains **{df['Primary Use'].nunique()} "
        "primary-use categories**:"
    )

    st.dataframe(
        primary_counts.rename("Number of Users").to_frame(),
        use_container_width=True
    )

    st.write(
        "The KMeans model identifies two user clusters. Comparing "
        "their average feature values helps highlight differences "
        "in smartphone usage and e-commerce spending."
    )

    # 6. Limitations
    st.markdown("### ⚠️ Limitations")

    st.markdown(
        """
        - The Random Forest classifier has limited predictive
          performance, so predictions should be interpreted cautiously.
        - The KMeans clusters overlap, and the silhouette score
          indicates weak separation between groups.
        - Cluster membership describes similarities in the selected
          numerical features; it does not establish users' motivations.
        - Findings describe this dataset and should not automatically
          be generalised to all smartphone users in India.
        """
    )

    # 7. Project conclusion
    st.markdown("### ✅ Conclusion")

    st.write(
        "This project demonstrates an end-to-end data science workflow "
        "combining data exploration, feature engineering, supervised "
        "learning, unsupervised clustering and interactive visualisation. "
        "It also illustrates the importance of evaluating model "
        "performance and communicating analytical limitations."
    )
