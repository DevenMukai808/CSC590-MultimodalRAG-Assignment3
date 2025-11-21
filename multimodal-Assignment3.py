import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")


# create a chromadb object
chroma_client = chromadb.PersistentClient(path="./data/chroma.db")

# instantiate image loader
image_loader = ImageLoader()

# instantiate multimodal embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# create the collection, - vector database
collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# add images to the collection add() or update() method
collection.add(
    ids=["0", "1"],
    uris=["./images/lion.jpg", "./images/tiger.jpg"],
    metadatas=[{"category": "animal"}, {"category": "animal"}],  # metadata - optional
)

# check the count of the collection
# print(collection.count())  # res: 2

# Use .add() to add a new record or .update() to update existing record
# on first run add() is used, on subsequent runs update() is used
collection.add(
    ids=[
        "E23",
        "E25",
        "E33",
        "Lion",
        "Tiger",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
        "S10",
    ],
    uris=[
        "./images/E23-2.jpg",
        "./images/E25-2.jpg",
        "./images/E33-2.jpg",
        "./images/lion.jpg",
        "./images/tiger.jpg",
        "./images/archery.jpg",
        "./images/baseball.jpg",
        "./images/basketball.jpg",
        "./images/bowling.jpg",
        "./images/football.jpg",
        "./images/golf.jpg",
        "./images/hockey.jpg",
        "./images/surfing.jpg",
        "./images/tennis.jpg",
        "./images/volleyball.jpg",
    ],
    metadatas=[
        {
            "item_id": "E23",
            "category": "food",
            "item_name": "Braised Fried Tofu with Greens",
        },
        {
            "item_id": "E25",
            "category": "food",
            "item_name": "Sauteed Assorted Vegetables",
        },
        {"item_id": "E33", "category": "food", "item_name": "Kung Pao Tofu"},
        {"item_id": "Lion", "category": "animal", "item_name": "Lion"},
        {"item_id": "Tiger", "category": "animal", "item_name": "Tiger"},
        {"item_id": "S1", "category": "sport", "item_name": "Archery"},
        {"item_id": "S2", "category": "sport", "item_name": "Baseball"},
        {"item_id": "S3", "category": "sport", "item_name": "Basketball"},
        {"item_id": "S4", "category": "sport", "item_name": "Bowling"},
        {"item_id": "S5", "category": "sport", "item_name": "Football"},
        {"item_id": "S6", "category": "sport", "item_name": "Golf"},
        {"item_id": "S7", "category": "sport", "item_name": "Hockey"},
        {"item_id": "S8", "category": "sport", "item_name": "Surfing"},
        {"item_id": "S9", "category": "sport", "item_name": "Tennis"},
        {"item_id": "S10", "category": "sport", "item_name": "Volleyball"},
    ],
)


# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict) -> None:
    result_count = len(query_results["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}"
            )

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()


# It is possible to submit multiple queries at the same time, just add to the list.
query_texts = ["sports, lion"]

# Query vector db - return 2 results
query_results = collection.query(
    query_texts=query_texts,
    n_results=2,
    include=["documents", "distances", "metadatas", "data", "uris"],
    # where={"category": "animal"}, # filter by metadata - optional - first run remove this
)

print_query_results(query_texts, query_results)
