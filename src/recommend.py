from reader import Reader
from cf import collab_filter

if __name__ == '__main__':
    db = Reader()
    user, book, review, user_to_review, book_to_review = db.get_full_data()
    book_to_review_train, book_to_review_test, user_to_review_train, user_to_review_test = db.get_parsed_data()

    # test corr_sim
    collab_filter(user_to_review_train, book_to_review_train, review, user_to_review_test, book_to_review_test)