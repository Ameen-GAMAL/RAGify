import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from datetime import datetime

class VectorDatabase:
    def __init__(self, embedding_model='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.responses = []
        self.dimension = 384
        
    def create_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def add_documents(self, texts, responses=None):
        embeddings = self.model.encode(texts)
        if self.index is None:
            self.create_index()
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.index.add(embeddings)
        self.documents.extend(texts)
        if responses:
            self.responses.extend(responses)
        else:
            self.responses.extend([None] * len(texts))
        
    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    'question': self.documents[idx],
                    'answer': self.responses[idx] if idx < len(self.responses) else None,
                    'score': float(distance),
                    'index': int(idx)
                })
        return results
    
    def save(self, path='ecommerce_db.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'responses': self.responses,
                'embeddings': self.embeddings,
                'dimension': self.dimension
            }, f)
        if self.index:
            faiss.write_index(self.index, 'ecommerce_faiss_index.bin')
            
    def load(self, path='ecommerce_db.pkl'):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.responses = data['responses']
                self.embeddings = data['embeddings']
                self.dimension = data['dimension']
            if os.path.exists('ecommerce_faiss_index.bin'):
                self.index = faiss.read_index('ecommerce_faiss_index.bin')

class SelfLearningLayer:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.feedback_log = []
        self.query_history = []
        
    def add_feedback(self, query, retrieved_docs, selected_doc_index, feedback_score):
        self.feedback_log.append({
            'timestamp': datetime.now(),
            'query': query,
            'retrieved_docs': len(retrieved_docs),
            'selected_doc': selected_doc_index,
            'feedback': feedback_score
        })
        
    def add_query_to_history(self, query):
        self.query_history.append({
            'timestamp': datetime.now(),
            'query': query
        })

class EcommerceRAGSystem:
    def __init__(self, csv_path=None):
        self.vector_db = VectorDatabase()
        self.learning_layer = SelfLearningLayer(self.vector_db)
        
        if csv_path and os.path.exists(csv_path):
            self.load_from_csv(csv_path)
        else:
            self.load_default_data()
        
    def load_default_data(self):
        default_questions = [
            "How can I create an account?",
            "What payment methods do you accept?",
            "How can I track my order?",
            "What is your return policy?",
            "Can I cancel my order?"
        ]
        
        default_answers = [
            "To create an account, click on the 'Sign Up' button and follow the instructions.",
            "We accept major credit cards, debit cards, and PayPal.",
            "You can track your order by logging into your account and going to 'Order History'.",
            "Our return policy allows returns within 30 days of purchase.",
            "You can cancel your order if it hasn't been shipped yet. Contact customer support."
        ]
        
        if not self.vector_db.documents:
            self.vector_db.add_documents(default_questions, default_answers)
            self.vector_db.save()
        else:
            self.vector_db.load()
    
    def load_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            if 'prompt' in df.columns and 'response' in df.columns:
                questions = df['prompt'].tolist()
                answers = df['response'].tolist()
                
                self.vector_db.add_documents(questions, answers)
                self.vector_db.save()
                return True, f"Loaded {len(questions)} questions from CSV file!"
            else:
                return False, "CSV file must contain 'prompt' and 'response' columns"
                
        except Exception as e:
            return False, f"Error loading CSV file: {e}"
    
    def retrieve(self, query, k=5):
        self.learning_layer.add_query_to_history(query)
        return self.vector_db.search(query, k)
    
    def generate_response(self, query, retrieved_docs):
        if not retrieved_docs:
            return "Sorry, I couldn't find a suitable answer to your question. Please contact customer support."
        
        best_match = retrieved_docs[0]
        
        if best_match['answer']:
            return best_match['answer']
        
        context = "\n".join([doc['question'] for doc in retrieved_docs[:3]])
        
        prompt = f"""Based on the following similar questions, answer the inquiry:

Similar questions:
{context}

Inquiry: {query}

Answer:"""
        
        return self.enhance_answer(prompt, query, retrieved_docs)
    
    def enhance_answer(self, prompt, query, retrieved_docs):
        answers = [doc['answer'] for doc in retrieved_docs if doc['answer']]
        
        if not answers:
            return "Sorry, I don't have a specific answer for your question. Please contact customer support for assistance."
        
        combined_answer = answers[0]
        
        if len(answers) > 1:
            additional_info = "\n\nAdditional information:\n" + "\n".join([f"• {ans[:100]}..." for ans in answers[1:3]])
            combined_answer += additional_info
        
        return combined_answer
    
    def add_new_qna(self, question, answer):
        self.vector_db.add_documents([question], [answer])
        self.vector_db.save()
        return True