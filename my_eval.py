
import chromadb

# ==========================================
# STEP 1: CREATE CHROMADB DATABASE
# ==========================================

client = chromadb.PersistentClient(
    path="./my_eval_db"
)

collection = client.get_or_create_collection(
    name="search_evaluation",
    configuration={"hnsw": {"space": "cosine"}}
)

# ==========================================
# STEP 2: CREATE 12 DOCUMENTS
# ==========================================

documents = {
    # API
    "doc_jwt": (
        "JWT tokens authenticate users in REST APIs "
        "using signed access tokens."
    ),
    "doc_pydantic": (
        "Pydantic validates incoming request data "
        "in FastAPI applications."
    ),
    "doc_cors": (
        "CORS allows frontend applications to "
        "make cross-origin requests to APIs."
    ),

    # FRONTEND
    "doc_session": (
        "Streamlit session state preserves values "
        "between script reruns."
    ),
    "doc_flex": (
        "CSS Flexbox arranges webpage elements "
        "in rows and columns."
    ),
    "doc_dom": (
        "JavaScript uses the DOM to modify "
        "HTML elements and page content."
    ),

    # AI
    "doc_embed": (
        "Embeddings convert text into numerical "
        "vectors that capture semantic meaning."
    ),
    "doc_cosine": (
        "Cosine similarity compares the direction "
        "of two embedding vectors."
    ),
    "doc_chunk": (
        "Chunking divides long documents into "
        "smaller searchable sections."
    ),

    # DATABASE
    "doc_chroma": (
        "ChromaDB stores embeddings and supports "
        "semantic similarity search."
    ),
    "doc_sql": (
        "SQL databases store information in "
        "tables with rows and columns."
    ),
    "doc_orm": (
        "SQLAlchemy maps Python classes to "
        "relational database tables."
    )
}

# ==========================================
# STEP 3: STORE DOCUMENTS
# ==========================================

doc_ids = list(documents.keys())
doc_texts = list(documents.values())

collection.upsert(
    ids=doc_ids,
    documents=doc_texts
)

print("Documents stored:", collection.count())

# ==========================================
# STEP 4: CREATE EVALUATION SET
# ==========================================

eval_set = [
    {
        "query": "How do I authenticate users in an API?",
        "relevant": ["doc_jwt"]
    },
    {
        "query": "How does FastAPI check user input?",
        "relevant": ["doc_pydantic"]
    },
    {
        "query": "How can Streamlit remember values?",
        "relevant": ["doc_session"]
    },
    {
        "query": "How does AI compare text meanings?",
        "relevant": ["doc_embed", "doc_cosine"]
    },
    {
        "query": "How can I organize information in tables?",
        "relevant": ["doc_sql", "doc_orm"]
    },
    {
        "query": "How do I prepare documents for semantic search?",
        "relevant": ["doc_chunk", "doc_chroma"]
    }
]

# ==========================================
# STEP 5: SEARCH FUNCTION
# ==========================================

def search(query, threshold=0.3, n_results=3):

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["distances", "documents"]
    )

    matches = []

    for doc_id, distance in zip(
        results["ids"][0],
        results["distances"][0]
    ):

        # Cosine similarity = 1 - cosine distance
        similarity = 1 - distance

        if similarity >= threshold:
            matches.append((doc_id, similarity))

    return matches


# ==========================================
# STEP 6: EVALUATION FUNCTION
# ==========================================

def evaluate(threshold=0.3, n_results=3):

    total_precision = 0
    total_recall = 0
    details = []

    print(
        f"\n=== Evaluation at threshold={threshold}, "
        f"n_results={n_results} ==="
    )

    for i, item in enumerate(eval_set, 1):

        query = item["query"]
        relevant = set(item["relevant"])

        results = search(
            query,
            threshold,
            n_results
        )

        returned = {
            doc_id for doc_id, score in results
        }

        correct = returned & relevant

        precision = (
            len(correct) / len(returned)
            if returned else 0
        )

        recall = (
            len(correct) / len(relevant)
            if relevant else 0
        )

        total_precision += precision
        total_recall += recall

        details.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "missed": sorted(relevant - returned)
        })

        print(
            f"\nQuery {i}: "
            f"P={precision:.1%} "
            f"R={recall:.1%}"
        )

        print("Question:", query)
        print("Expected:", sorted(relevant))

        print("Retrieved:")

        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")

        if relevant - returned:
            print("Missed:", sorted(relevant - returned))

    avg_precision = total_precision / len(eval_set)
    avg_recall = total_recall / len(eval_set)

    print("\n" + "=" * 55)
    print(
        f"AVERAGE: "
        f"P={avg_precision:.1%} "
        f"R={avg_recall:.1%}"
    )

    return avg_precision, avg_recall, details


# ==========================================
# STEP 7: COMPARE 3 SETTINGS
# ==========================================

settings = [
    (0.20, 3),
    (0.35, 3),
    (0.50, 5)
]

comparison = []

for threshold, n_results in settings:

    precision, recall, details = evaluate(
        threshold,
        n_results
    )

    comparison.append({
        "threshold": threshold,
        "n_results": n_results,
        "precision": precision,
        "recall": recall,
        "details": details
    })


# ==========================================
# STEP 8: DISPLAY COMPARISON
# ==========================================

print("\n=== COMPARISON ===")

for item in comparison:
    print(
        f"Threshold={item['threshold']:.2f}, "
        f"n_results={item['n_results']} | "
        f"Precision={item['precision']:.1%}, "
        f"Recall={item['recall']:.1%}"
    )


# ==========================================
# STEP 9: AUTOMATIC ANALYSIS
# ==========================================

print("\n=== ANALYSIS ===")

for i, test in enumerate(eval_set):

    query_results = [
        result["details"][i]
        for result in comparison
    ]

    best_recall = max(
        result["recall"]
        for result in query_results
    )

    best_precision = max(
        result["precision"]
        for result in query_results
    )

    print(f"\nQuery {i + 1}: {test['query']}")

    if best_precision >= 0.6 and best_recall >= 0.5:
        print("Good performance on at least one setting.")
    else:
        print("Search quality needs improvement.")

    missed_all = set(test["relevant"])

    for result in query_results:
        missed_all &= set(result["missed"])

    if missed_all:
        print("Consistently missed:", sorted(missed_all))
        print(
            "Suggestion: Improve document wording, "
            "review relevance labels, or adjust retrieval."
        )
    else:
        print("No relevant document was missed in every setting.")


# ==========================================
# STEP 10: FINAL SUMMARY
# ==========================================

print("\n=== FINAL SUMMARY ===")

for item in comparison:
    print(
        f"Threshold {item['threshold']:.2f}: "
        f"Precision {item['precision']:.1%}, "
        f"Recall {item['recall']:.1%}"
    )

print("\nSearch evaluation completed!")
