import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time
import os


# Initialize session state variables at the very beginning of your app
def init_session_state():
    """Initialize all session state variables used in the app."""
    if 'nltk_resources_downloaded' not in st.session_state:
        st.session_state['nltk_resources_downloaded'] = False

# Call this function at the very beginning of your main function
init_session_state()

def setup_nltk_resources():
    """
    Setup NLTK resources with proper waiting and verification.
    This function must be called before any text processing.
    """
    # Initialize session state if not already done
    if 'nltk_resources_downloaded' not in st.session_state:
        st.session_state['nltk_resources_downloaded'] = False
    
    # Skip if already downloaded in this session
    if st.session_state['nltk_resources_downloaded']:
        return True
    
    # Show a loading message in the sidebar instead of the main area
    with st.sidebar:
        status_placeholder = st.empty()
        status_placeholder.info("Téléchargement des ressources linguistiques nécessaires...")
        
        # Create NLTK data directory
        nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Add to NLTK's search path
        nltk.data.path.insert(0, nltk_data_dir)
        
        # Define resources to download
        resources = [
            'punkt',
            'stopwords',
            'wordnet',
        ]
        
        # Download resources
        success = True
        for resource in resources:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                # Wait for download to complete
                time.sleep(2)
            except Exception as e:
                status_placeholder.error(f"Error downloading {resource}: {str(e)}")
                success = False
        
        # Wait for downloads to complete
        time.sleep(3)  # Additional waiting time
        
        # Mark as downloaded
        st.session_state['nltk_resources_downloaded'] = success
        
        # Update status message
        if success:
            status_placeholder.success("Ressources linguistiques téléchargées avec succès!")
        
        return success
# Modify your preprocess_text function to include fallback mechanisms
def preprocess_text(text, lang='french'):
    """
    Preprocess text with robust fallback mechanisms.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Conversion en minuscules
    text = text.lower()
    
    # Suppression des caractères spéciaux et chiffres
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenization with fallback
    try:
        # First attempt: standard tokenization
        tokens = word_tokenize(text, language=lang)
    except LookupError:
        try:
            # Second attempt: try without language specification
            tokens = word_tokenize(text)
        except LookupError:
            # Final fallback: simple space splitting
            tokens = text.split()
    
    # Stopwords with fallback
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words(lang))
    except (LookupError, OSError):
        # Fallback French stopwords
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'il', 'elle', 
            'je', 'tu', 'nous', 'vous', 'ils', 'elles', 'à', 'de', 'ce', 'cette',
            'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes', 'ses',
            'pour', 'par', 'en', 'sur', 'sous', 'dans', 'avec', 'sans'
        }
    
    # Add domain-specific stopwords
    stop_words.update(['hotel', 'chambre', 'paris', 'jour', 'nuit', 'bien', 
                      'très', 'plus', 'moins', 'tout', 'petit', 'grande'])
    
    # Filter out stopwords and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatization with fallback
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except (LookupError, OSError):
        # No lemmatization, just return the tokens
        pass
    
    return tokens
# Extraction des mots les plus fréquents
def get_frequent_words(texts, top_n=20):
    all_tokens = []
    for text in texts:
        tokens = preprocess_text(text)
        all_tokens.extend(tokens)
    
    # Comptage des occurrences
    word_counts = Counter(all_tokens)
    
    # Retourne les n mots les plus fréquents
    return word_counts.most_common(top_n)

# Création d'un nuage de mots
def create_wordcloud(word_counts, title):
    if not word_counts:
        return None
    
    wc_data = {word: count for word, count in word_counts}
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1
    ).generate_from_frequencies(wc_data)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    return fig

# Extraction des thématiques (topics) via LDA
def extract_topics(texts, num_topics=5, num_words=10):
    if not texts or all(not isinstance(text, str) or not text.strip() for text in texts):
        return [], []
    
    # Prétraitement des textes
    processed_texts = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            tokens = preprocess_text(text)
            processed_texts.append(' '.join(tokens))
    
    if not processed_texts:
        return [], []
    
    # Vectorisation du texte
    vectorizer = CountVectorizer(max_features=1000)
    dtm = vectorizer.fit_transform(processed_texts)
    
    # Application de LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # Extraction des mots par thématique
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    topic_keywords = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(f"Thématique {topic_idx+1}")
        topic_keywords.append(', '.join(top_words))
    
    return topics, topic_keywords

# Analyse des sentiments et extraction de KPIs
def analyze_sentiment_and_extract_kpis(positive_df, negative_df):
    setup_nltk_resources()
    # Vérifier si les dataframes sont vides
    if positive_df.empty and negative_df.empty:
        st.error("Aucune donnée disponible pour l'analyse des sentiments.")
        return
    
    # Préparation des données
    positive_comments = []
    if not positive_df.empty and 'Positive Comment' in positive_df.columns:
        positive_comments = positive_df['Positive Comment'].dropna().tolist()
    
    negative_comments = []
    if not negative_df.empty and 'Negative Comment' in negative_df.columns:
        negative_comments = negative_df['Negative Comment'].dropna().tolist()
    
    # KPI 1: Nombre total de commentaires positifs et négatifs
    total_positive = len(positive_comments)
    total_negative = len(negative_comments)
    total_comments = total_positive + total_negative
    
    # Affichage des KPIs de base
    st.header("Analyse des Sentiments - KPIs Principaux")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total des commentaires", total_comments)
    with col2:
        st.metric("Commentaires positifs", total_positive, 
                 f"{round((total_positive/total_comments)*100) if total_comments > 0 else 0}%")
    with col3:
        st.metric("Commentaires négatifs", total_negative,
                 f"{round((total_negative/total_comments)*100) if total_comments > 0 else 0}%")
    
    # Graphique du sentiment
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positif', 'Négatif'],
        'Nombre': [total_positive, total_negative]
    })
    
    fig = px.pie(sentiment_data, values='Nombre', names='Sentiment', 
                title='Répartition des Sentiments',
                color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)
    
    # KPI 2: Mots les plus fréquents par sentiment
    st.header("Analyse des Mots les Plus Fréquents")
    
    tab1, tab2 = st.tabs(["Commentaires Positifs", "Commentaires Négatifs"])
    
    with tab1:
        if positive_comments:
            pos_freq_words = get_frequent_words(positive_comments, top_n=20)
            
            # Graphique à barres pour les mots positifs
            pos_words_df = pd.DataFrame(pos_freq_words, columns=['Mot', 'Fréquence'])
            fig_pos_bar = px.bar(pos_words_df, x='Mot', y='Fréquence', 
                                title='Top 20 des mots dans les commentaires positifs',
                                color='Fréquence', color_continuous_scale='Viridis')
            st.plotly_chart(fig_pos_bar, use_container_width=True)
            
            # Nuage de mots positifs
            st.subheader("Nuage de mots - Commentaires Positifs")
            fig_pos_wc = create_wordcloud(pos_freq_words, "Mots Fréquents dans les Commentaires Positifs")
            if fig_pos_wc:
                st.pyplot(fig_pos_wc)
        else:
            st.info("Aucun commentaire positif disponible pour l'analyse.")
    
    with tab2:
        if negative_comments:
            neg_freq_words = get_frequent_words(negative_comments, top_n=20)
            
            # Graphique à barres pour les mots négatifs
            neg_words_df = pd.DataFrame(neg_freq_words, columns=['Mot', 'Fréquence'])
            fig_neg_bar = px.bar(neg_words_df, x='Mot', y='Fréquence', 
                                title='Top 20 des mots dans les commentaires négatifs',
                                color='Fréquence', color_continuous_scale='Reds')
            st.plotly_chart(fig_neg_bar, use_container_width=True)
            
            # Nuage de mots négatifs
            st.subheader("Nuage de mots - Commentaires Négatifs")
            fig_neg_wc = create_wordcloud(neg_freq_words, "Mots Fréquents dans les Commentaires Négatifs")
            if fig_neg_wc:
                st.pyplot(fig_neg_wc)
        else:
            st.info("Aucun commentaire négatif disponible pour l'analyse.")
    
    # KPI 3: Extraction des thématiques
    st.header("Analyse des Thématiques")
    
    num_topics = st.slider("Nombre de thématiques à extraire", min_value=2, max_value=10, value=5)
    
    tab3, tab4 = st.tabs(["Thématiques Positives", "Thématiques Négatives"])
    
    with tab3:
        if positive_comments:
            pos_topics, pos_keywords = extract_topics(positive_comments, num_topics=num_topics)
            
            if pos_topics:
                pos_topics_df = pd.DataFrame({
                    'Thématique': pos_topics,
                    'Mots-clés': pos_keywords
                })
                st.dataframe(pos_topics_df, use_container_width=True)
                
                # Visualisation des thématiques positives
                fig_pos_topics = go.Figure(data=[go.Table(
                    header=dict(
                        values=['<b>Thématique</b>', '<b>Mots-clés</b>'],
                        line_color='white', fill_color='#00CC96',
                        align='center', font=dict(color='white', size=14)
                    ),
                    cells=dict(
                        values=[pos_topics_df['Thématique'], pos_topics_df['Mots-clés']],
                        line_color='white', fill_color=['rgba(0, 204, 150, 0.2)', 'white'],
                        align='left', font=dict(color='darkslategray', size=12)
                    ))
                ])
                
                fig_pos_topics.update_layout(
                    title='Thématiques extraites des commentaires positifs',
                    width=800
                )
                st.plotly_chart(fig_pos_topics, use_container_width=True)
            else:
                st.info("Impossible d'extraire des thématiques des commentaires positifs.")
        else:
            st.info("Aucun commentaire positif disponible pour l'analyse.")
    
    with tab4:
        if negative_comments:
            neg_topics, neg_keywords = extract_topics(negative_comments, num_topics=num_topics)
            
            if neg_topics:
                neg_topics_df = pd.DataFrame({
                    'Thématique': neg_topics,
                    'Mots-clés': neg_keywords
                })
                st.dataframe(neg_topics_df, use_container_width=True)
                
                # Visualisation des thématiques négatives
                fig_neg_topics = go.Figure(data=[go.Table(
                    header=dict(
                        values=['<b>Thématique</b>', '<b>Mots-clés</b>'],
                        line_color='white', fill_color='#EF553B',
                        align='center', font=dict(color='white', size=14)
                    ),
                    cells=dict(
                        values=[neg_topics_df['Thématique'], neg_topics_df['Mots-clés']],
                        line_color='white', fill_color=['rgba(239, 85, 59, 0.2)', 'white'],
                        align='left', font=dict(color='darkslategray', size=12)
                    ))
                ])
                
                fig_neg_topics.update_layout(
                    title='Thématiques extraites des commentaires négatifs',
                    width=800
                )
                st.plotly_chart(fig_neg_topics, use_container_width=True)
            else:
                st.info("Impossible d'extraire des thématiques des commentaires négatifs.")
        else:
            st.info("Aucun commentaire négatif disponible pour l'analyse.")
    
    # KPI 4: Comparaison des mots communs entre positifs et négatifs
    if positive_comments and negative_comments:
        st.header("Comparaison des Sentiments")
        
        pos_words_dict = dict(get_frequent_words(positive_comments, top_n=50))
        neg_words_dict = dict(get_frequent_words(negative_comments, top_n=50))
        
        common_words = set(pos_words_dict.keys()).intersection(set(neg_words_dict.keys()))
        
        if common_words:
            comparison_data = []
            for word in common_words:
                comparison_data.append({
                    'Mot': word,
                    'Fréquence_Positive': pos_words_dict.get(word, 0),
                    'Fréquence_Négative': neg_words_dict.get(word, 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values(by=['Fréquence_Positive', 'Fréquence_Négative'], 
                                                     ascending=False)
            
            # Graphique de comparaison
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comparison_df['Mot'],
                y=comparison_df['Fréquence_Positive'],
                name='Commentaires Positifs',
                marker_color='#00CC96'
            ))
            fig_comp.add_trace(go.Bar(
                x=comparison_df['Mot'],
                y=comparison_df['Fréquence_Négative'],
                name='Commentaires Négatifs',
                marker_color='#EF553B'
            ))
            
            fig_comp.update_layout(
                title='Comparaison des mots communs entre commentaires positifs et négatifs',
                xaxis_title='Mot',
                yaxis_title='Fréquence',
                barmode='group',
                height=600
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tableau de comparaison
            st.subheader("Tableau de comparaison des mots communs")
            comparison_display = comparison_df.copy()
            comparison_display['Ratio_Pos_Neg'] = comparison_display['Fréquence_Positive'] / comparison_display['Fréquence_Négative']
            comparison_display = comparison_display.round(2)
            st.dataframe(comparison_display, use_container_width=True)
        else:
            st.info("Aucun mot commun trouvé entre les commentaires positifs et négatifs.")
