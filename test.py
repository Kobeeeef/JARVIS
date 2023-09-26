
while True:
    i = input("> ")
    most_similar_score, most_similar_text = find_most_similar(i, titles, descriptions)

    # Check if the most similar text is a description, if so, display the corresponding title
    most_similar_title = most_similar_text
    if most_similar_text in descriptions:
        index = descriptions.index(most_similar_text)
        most_similar_title = titles[index]

    # Check if the similarity score is below a certain threshold to consider it "None"
    min_similarity_threshold = 40
    if most_similar_score < min_similarity_threshold:
        most_similar_title = "None"
    print("Most similar title:", most_similar_title)
