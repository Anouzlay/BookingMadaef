import pandas as pd
import numpy as np
import re
from collections import Counter
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import openai
import time



# Fonction pour traiter le texte avec ChatGPT
def preprocess_text_with_chatgpt(text):
    if not isinstance(text, str) or not text.strip():
        return []
    # Créer un prompt pour ChatGPT
    system_prompt = """
    Tu es un assistant spécialisé dans le traitement de texte. Je vais te donner un texte et tu dois :
    1. Détecter automatiquement la langue du texte
    2. Supprimer tous les mots vides (stopwords) spécifiques à cette langue
    3. Supprimer tous les chiffres et caractères spéciaux
    4. Lemmatiser les mots (réduire les mots à leur forme de base)
    5. Ignorer les mots spécifiques au domaine hôtelier (hotel, chambre, room, etc.)
    6. Ne garder que les mots de plus de 2 caractères
    7. Retourner uniquement une liste des mots significatifs, sans aucune explication

    Format de réponse attendu: mot1, mot2, mot3...
    """
    
    user_prompt = f"Voici le texte à traiter : {text}"
    
    try:
        # Appel à l'API OpenAI avec gestion des erreurs et des tentatives
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",  # ou "gpt-4" si vous avez accès
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Température basse pour des résultats plus déterministes
                    max_tokens=500
                )
                
                # Extraire la réponse
                processed_text = response.choices[0].message.content.strip()
                
                # Traiter la réponse pour obtenir une liste de mots
                processed_words = [word.strip() for word in processed_text.split(',')]
                return processed_words
            
            except (openai.APIError, openai.RateLimitError) as e:
                if attempt < max_retries - 1:
                    st.warning(f"Erreur API OpenAI: {e}. Nouvelle tentative dans {retry_delay} secondes...")
                    time.sleep(retry_delay * (attempt + 1))  # Délai exponentiel
                else:
                    st.error(f"Erreur API OpenAI après {max_retries} tentatives: {e}")
                    return []
            except Exception as e:
                st.error(f"Erreur inattendue: {e}")
                return []
    
    except Exception as e:
        st.error(f"Erreur lors du traitement avec ChatGPT: {e}")
        return []

# Extraction des mots les plus fréquents
def get_frequent_words(texts, top_n=20):
    all_tokens = []
    
    # Barre de progression
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i, text in enumerate(texts):
        # Mettre à jour la progression
        progress = int((i + 1) / len(texts) * 100)
        progress_bar.progress(progress)
        progress_text.text(f"Traitement du texte {i+1}/{len(texts)}")
        
        tokens = preprocess_text_with_chatgpt(text)
        all_tokens.extend(tokens)
    
    # Effacer la barre de progression
    progress_bar.empty()
    progress_text.empty()
    
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

# Extraction des thématiques avec ChatGPT
def extract_topics_with_chatgpt(texts, num_topics=5):
    if not texts or all(not isinstance(text, str) or not text.strip() for text in texts):
        return [], []
    
    # Limiter le nombre de textes pour éviter de dépasser les limites de l'API
    max_texts = 150
    if len(texts) > max_texts:
        st.warning(f"Nombre de commentaires limité à {max_texts} pour l'analyse des thématiques.")
        sample_texts = texts[:max_texts]
    else:
        sample_texts = texts
    
    # Combiner les textes en un seul document, en limitant la taille
    combined_text = " ".join([text for text in sample_texts if isinstance(text, str) and text.strip()])
    
    # Limiter la taille du texte combiné pour respecter les limites de l'API
    max_chars = 12000  # Ajustez cette valeur selon les limites de votre modèle
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
    
    if not combined_text:
        return [], []
    
    # Créer un prompt pour ChatGPT
    system_prompt = f"""
    Tu es un expert en analyse de texte et en extraction de thématiques. Je vais te donner un ensemble de commentaires, et tu vas identifier {num_topics} thématiques principales.

    Pour chaque thématique :
    1. Donne un titre clair et concis à la thématique
    2. Liste exactement 8 mots-clés associés à cette thématique, séparés par des virgules

    Format de réponse EXACT (sans explications supplémentaires) :
    Thématique 1: [Titre]
    Mots-clés: mot1, mot2, mot3, mot4, mot5, mot6, mot7, mot8

    Thématique 2: [Titre]
    Mots-clés: mot1, mot2, mot3, mot4, mot5, mot6, mot7, mot8

    (et ainsi de suite pour toutes les thématiques demandées)
    """
    
    user_prompt = f"Voici les commentaires à analyser pour extraire {num_topics} thématiques : {combined_text}"
    
    try:
        # Appel à l'API OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extraire la réponse
        topics_text = response.choices[0].message.content.strip()
        
        # Diviser la réponse par thématique
        topic_sections = topics_text.split('\n\n')
        
        topics = []
        topic_keywords = []
        
        for section in topic_sections:
            lines = section.strip().split('\n')
            if len(lines) >= 2 and lines[0].startswith('Thématique'):
                # Extraire le titre
                title_parts = lines[0].split(':', 1)
                if len(title_parts) > 1:
                    topic_title = title_parts[1].strip()
                    topics.append(topic_title)
                
                # Extraire les mots-clés
                for line in lines[1:]:
                    if line.startswith('Mots-clés:'):
                        keywords = line.split(':', 1)[1].strip()
                        topic_keywords.append(keywords)
                        break
        
        return topics, topic_keywords
    
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des thématiques avec ChatGPT: {e}")
        return [], []
def analyze_sentiment_and_extract_kpis(positive_df, negative_df , open_ai_key):
    openai.api_key=open_ai_key
    # Vérifier si les dataframes sont vides
    if positive_df.empty and negative_df.empty:
        st.error("Aucune donnée disponible pour l'analyse des sentiments.")
        return
    
    st.info("Analyse des Commentaires avec IA")
    positive_comments = []
    if not positive_df.empty and 'Positive Comment' in positive_df.columns:
        positive_comments = positive_df['Positive Comment'].dropna().tolist()
    
    negative_comments = []
    if not negative_df.empty and 'Negative Comment' in negative_df.columns:
        negative_comments = negative_df['Negative Comment'].dropna().tolist()

    total_positive = len(positive_comments)
    total_negative = len(negative_comments)
    total_comments = total_positive + total_negative

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

    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positif', 'Négatif'],
        'Nombre': [total_positive, total_negative]
    })
    
    fig = px.pie(sentiment_data, values='Nombre', names='Sentiment', 
                title='Répartition des Sentiments',
                color_discrete_sequence=['#00CC96', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)

    max_slider_value = max(1, min(200, max(total_positive, total_negative)))

    if max_slider_value == 1:
        max_slider_value = 2
    default_value = min(50, max_slider_value)
    step_value = 1 if max_slider_value < 20 else 10
    _slider_container = st.sidebar.empty()
    max_comments = _slider_container.slider(
        "Nombre maximum de commentaires à analyser",
        min_value=1,
        max_value=max_slider_value,
        value=default_value,
        step=step_value
    )
    _slider_container.empty()
    limited_positive_comments = []
    limited_negative_comments = []
    
    if positive_comments:
        limited_positive_comments = positive_comments[:min(max_comments, len(positive_comments))]
        if len(limited_positive_comments) < len(positive_comments):
            st.sidebar.info(f"Analyse limitée aux {len(limited_positive_comments)} premiers commentaires positifs")
    
    if negative_comments:
        limited_negative_comments = negative_comments[:min(max_comments, len(negative_comments))]
        if len(limited_negative_comments) < len(negative_comments):
            st.sidebar.info(f"Analyse limitée aux {len(limited_negative_comments)} premiers commentaires négatifs")
    st.header("Analyse des Mots les Plus Fréquents")
    
    tab1, tab2 = st.tabs(["Commentaires Positifs", "Commentaires Négatifs"])
    pos_freq_words = []
    neg_freq_words = []
    
    with tab1:
        if limited_positive_comments:
            
            with st.spinner("Analyse des mots fréquents dans les commentaires positifs..."):
                pos_freq_words = get_frequent_words(limited_positive_comments, top_n=20)
            
            # Graphique à barres pour les mots positifs
            if pos_freq_words:
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
                st.warning("Aucun mot significatif trouvé dans les commentaires positifs.")
        else:
            st.info("Aucun commentaire positif disponible pour l'analyse.")
    
    with tab2:
        if limited_negative_comments:
            st.info("Traitement du texte avec IA")
            
            with st.spinner("Analyse des mots fréquents dans les commentaires négatifs..."):
                neg_freq_words = get_frequent_words(limited_negative_comments, top_n=20)
            
            # Graphique à barres pour les mots négatifs
            if neg_freq_words:
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
                st.warning("Aucun mot significatif trouvé dans les commentaires négatifs.")
        else:
            st.info("Aucun commentaire négatif disponible pour l'analyse.")
    # KPI 3: Extraction des thématiques
    st.header("Analyse des Thématiques")
    num_topics = 5
    
    tab3, tab4 = st.tabs(["Thématiques Positives", "Thématiques Négatives"])
    with tab3:
        if limited_positive_comments:
            with st.spinner("Extraction des thématiques des commentaires positifs avec IA..."):
                pos_topics, pos_keywords = extract_topics_with_chatgpt(limited_positive_comments, num_topics=num_topics)
            
            if pos_topics:
                pos_topics_df = pd.DataFrame({
                    'Thématique': pos_topics,
                    'Mots-clés': pos_keywords
                })
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
        if limited_negative_comments:
            with st.spinner("Extraction des thématiques des commentaires négatifs avec IA..."):
                neg_topics, neg_keywords = extract_topics_with_chatgpt(limited_negative_comments, num_topics=num_topics)
            
            if neg_topics:
                neg_topics_df = pd.DataFrame({
                    'Thématique': neg_topics,
                    'Mots-clés': neg_keywords
                })
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
    
