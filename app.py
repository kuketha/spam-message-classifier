# ============================================================
# 📩 SPAM MESSAGE CLASSIFIER - STREAMLIT APP
# ============================================================
# This application predicts whether a message is:
#   1. SPAM  → unwanted/fraudulent/promotional message
#   2. HAM   → normal/not-spam message
#
# The application uses:
#   - Streamlit       → Web application interface
#   - Pickle          → Load trained ML model and vectorizer
#   - TF-IDF          → Convert text into numerical features
#   - Scikit-learn    → Machine Learning prediction
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

# Import Streamlit to create the web application
import streamlit as st

# Import pickle to load the saved ML model and TF-IDF vectorizer
import pickle


# ------------------------------------------------------------
# 2. CONFIGURE THE STREAMLIT PAGE
# ------------------------------------------------------------

# Configure the browser page title, icon and layout
st.set_page_config(
    page_title="Spam Message Classifier",  # Title shown in browser tab
    page_icon="📩",                         # Icon shown in browser tab
    layout="centered"                       # Keep application content centered
)


# ------------------------------------------------------------
# 3. ADD CUSTOM CSS
# ------------------------------------------------------------

# Streamlit allows us to add custom CSS using st.markdown().
#
# Here we change the main application background to WHITE.
# We also change text colors to dark colors so that
# the text is clearly visible on the white background.

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Main application background
       ------------------------------------------------------- */

    .stApp {
        background-color: white;
    }


    /* -------------------------------------------------------
       Main text color
       ------------------------------------------------------- */

    .stApp,
    .stMarkdown,
    p,
    label {
        color: #222222;
    }


    /* -------------------------------------------------------
       Text area styling
       ------------------------------------------------------- */

    textarea {
        background-color: white !important;
        color: #222222 !important;
        border: 1px solid #cccccc !important;
    }


    /* -------------------------------------------------------
       Text input placeholder color
       ------------------------------------------------------- */

    textarea::placeholder {
        color: #777777 !important;
    }


    /* -------------------------------------------------------
       Expander styling
       ------------------------------------------------------- */

    div[data-testid="stExpander"] {
        background-color: white;
        border: 1px solid #dddddd;
        border-radius: 8px;
    }


    /* -------------------------------------------------------
       Code/example message background
       ------------------------------------------------------- */

    code {
        background-color: #f5f5f5 !important;
        color: #222222 !important;
    }


    /* -------------------------------------------------------
       Caption text
       ------------------------------------------------------- */

    .stCaption {
        color: #666666 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# 4. LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ------------------------------------------------------------

# @st.cache_resource tells Streamlit to load the model only once.
#
# Without this, Streamlit could reload the model every time
# the user interacts with the application.
#
# Caching makes the application faster.

@st.cache_resource
def load_artifacts():

    # Open the saved spam classification model
    # "rb" means read the file in binary mode.
    with open("spam_model.pkl", "rb") as f:

        # Load the trained model from the pickle file
        model = pickle.load(f)


    # Open the saved TF-IDF vectorizer
    with open("tfidf_vectorizer.pkl", "rb") as f:

        # Load the vectorizer from the pickle file
        vectorizer = pickle.load(f)


    # Return both the model and vectorizer
    return model, vectorizer


# ------------------------------------------------------------
# 5. CALL THE FUNCTION TO LOAD THE MODEL
# ------------------------------------------------------------

# Load the trained model and TF-IDF vectorizer
model, vectorizer = load_artifacts()


# ------------------------------------------------------------
# 6. APPLICATION TITLE
# ------------------------------------------------------------

# Display the main title of the application
st.title("📩 Spam Message Classifier")


# ------------------------------------------------------------
# 7. APPLICATION DESCRIPTION
# ------------------------------------------------------------

# Display a short explanation below the title
st.write(
    "Paste an SMS/email message below and the model will "
    "predict whether it's *spam* or *ham* (not spam)."
)


# ------------------------------------------------------------
# 8. CREATE MESSAGE INPUT BOX
# ------------------------------------------------------------

# Create a large text box where the user can enter a message
message = st.text_area(
    "Enter your message:",

    # Set the height of the text area
    height=150,

    # Display an example inside the empty text box
    placeholder=(
        "e.g. Congratulations! You've won a free prize, "
        "click here to claim..."
    )
)


# ------------------------------------------------------------
# 9. CREATE PREDICT BUTTON
# ------------------------------------------------------------

# Create two columns.
#
# col1 will contain the Predict button.
# col2 provides empty space so that the button doesn't
# stretch across the entire page.

col1, col2 = st.columns([1, 3])


# Put the Predict button inside column 1
with col1:

    # Create the Predict button
    predict_clicked = st.button(
        "Predict",

        # Make this the primary Streamlit button
        type="primary",

        # Make the button use the available column width
        use_container_width=True
    )


# ------------------------------------------------------------
# 10. CHECK WHETHER USER CLICKED PREDICT
# ------------------------------------------------------------

# This block executes only when the user clicks Predict.
if predict_clicked:

    # --------------------------------------------------------
    # 11. CHECK WHETHER MESSAGE IS EMPTY
    # --------------------------------------------------------

    # strip() removes unnecessary spaces from the beginning
    # and end of the message.
    #
    # "not message.strip()" means the user didn't enter
    # any meaningful text.

    if not message.strip():

        # Display a warning message
        st.warning("Please enter a message first.")


    # --------------------------------------------------------
    # 12. PROCESS THE MESSAGE
    # --------------------------------------------------------

    else:

        # Convert the text message into TF-IDF numerical features.
        #
        # Machine Learning models cannot directly understand
        # raw text.
        #
        # TF-IDF converts words into numerical values.

        vec = vectorizer.transform([message])


        # ----------------------------------------------------
        # 13. MAKE THE PREDICTION
        # ----------------------------------------------------

        # Send the TF-IDF features to the trained model.
        #
        # [0] extracts the first prediction from the returned
        # array.

        prediction = model.predict(vec)[0]


        # ----------------------------------------------------
        # 14. CALCULATE CONFIDENCE / DECISION SCORE
        # ----------------------------------------------------

        # Create an empty confidence text.
        confidence_text = ""


        # ----------------------------------------------------
        # 15. CHECK FOR predict_proba()
        # ----------------------------------------------------

        # Some ML models such as LogisticRegression,
        # RandomForestClassifier, etc. support predict_proba().
        #
        # This gives probability values for each class.

        if hasattr(model, "predict_proba"):

            # Calculate probability for each class
            proba = model.predict_proba(vec)[0]

            # Get the highest probability and convert it
            # into a percentage.
            confidence_text = (
                f" (confidence: {max(proba) * 100:.1f}%)"
            )


        # ----------------------------------------------------
        # 16. CHECK FOR decision_function()
        # ----------------------------------------------------

        # LinearSVC does not normally provide predict_proba().
        #
        # Instead, LinearSVC provides decision_function().
        #
        # The decision score indicates which side of the
        # classification boundary the message falls on.

        elif hasattr(model, "decision_function"):

            # Calculate the decision score
            score = model.decision_function(vec)[0]

            # Display the decision score
            confidence_text = (
                f" (decision score: {score:.2f})"
            )


        # ----------------------------------------------------
        # 17. CHECK THE PREDICTION
        # ----------------------------------------------------

        # If the model returns 1,
        # we assume that 1 represents SPAM.

        if prediction == 1:

            # Display a red error-style message
            st.error(
                f"🚨 This message looks like *SPAM*"
                f"{confidence_text}"
            )


        # ----------------------------------------------------
        # 18. HAM / NOT SPAM RESULT
        # ----------------------------------------------------

        # If prediction is not 1,
        # we assume that it represents HAM.

        else:

            # Display a green success-style message
            st.success(
                f"✅ This message looks like *HAM* "
                f"(not spam){confidence_text}"
            )


# ------------------------------------------------------------
# 19. ADD A HORIZONTAL DIVIDER
# ------------------------------------------------------------

# Create a horizontal line to separate the prediction
# section from the examples section.
st.divider()


# ------------------------------------------------------------
# 20. CREATE EXAMPLE MESSAGE SECTION
# ------------------------------------------------------------

# st.expander() creates a collapsible section.
#
# The user can click "Try example messages" to see examples.

with st.expander("Try example messages"):

    # Create a list containing sample spam and ham messages.
    examples = [

        # Example 1 - SPAM
        "Congratulations! You've WON a $1000 gift card. "
        "Click here to claim now!!!",

        # Example 2 - HAM
        "Hey, are we still on for lunch tomorrow at 1pm?",

        # Example 3 - SPAM
        "URGENT: Your account has been suspended, "
        "verify immediately at this link",

        # Example 4 - HAM
        "Can you send me the report before end of day?",
    ]


    # --------------------------------------------------------
    # 21. DISPLAY EACH EXAMPLE
    # --------------------------------------------------------

    # Loop through every message in the examples list.
    for ex in examples:

        # Display each example as a code-style text box.
        st.code(
            ex,
            language=None
        )


# ------------------------------------------------------------
# 22. DISPLAY MODEL INFORMATION
# ------------------------------------------------------------

# Display information about the dataset and ML approach
# used to train the model.

st.caption(
    "Model: trained on the SMS Spam Collection dataset "
    "using TF-IDF + a scikit-learn classifier."
)


# ============================================================
# END OF APPLICATION
# ============================================================

