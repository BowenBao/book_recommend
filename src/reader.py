import random
import math

class Reader:
    def __init__(self):
        self.read_data()

    def read_data(self):
        userPath = '../data/userDB.txt'
        authorPath = '../data/authorDB.txt'
        reviewPath = '../data/reviewDB.txt' # reviewId bookId userId rating review
        bookPath = '../data/bookDB.txt'

        self.user = set()
        self.book = {}
        self.author = {}
        self.review = {}

        self.user_to_review = {}
        self.book_to_review = {}

        self.user_to_review_train = {}
        self.book_to_review_train = {}

        self.user_to_review_test = {}
        self.book_to_review_test = {}

        test_ratio = 0.1

        print('Start reading user...')
        with open(userPath, 'r') as userFile:
            for line in userFile:
                self.user.add(int(line.strip()))

        print('Start reading book...')
        with open(bookPath, 'r') as bookFile:
            for line in bookFile:
                vals = line.strip().split('\t')
                self.book[int(vals[0])] = vals[1]

        print('Start reading author...')
        with open(authorPath, 'r') as authorFile:
            for line in authorFile:
                vals = line.strip().split('\t')
                self.author[int(vals[0])] = vals[1]

        print('Start reading review...')
        with open(reviewPath, 'r') as reviewFile:
            for line in reviewFile:
                vals = line.strip().split('\t')

                reviewId = int(vals[0])

                bookId = int(vals[1])
                userId = int(vals[2])

                self.review[reviewId] = [int(vals[3]), bookId, userId, vals[4]]

                def checkExist(id, set, detail):
                    if id not in set:
                        print(detail)

                checkExist(bookId, self.book, '%d not found in books' % bookId)
                checkExist(userId, self.user, '%d not found in users' % userId)

                # update links from user/book to review
                if bookId in self.book:
                    if bookId not in self.book_to_review:
                        self.book_to_review[bookId] = []
                    self.book_to_review[bookId].append(reviewId)
                if userId in self.user:
                    if userId not in self.user_to_review:
                        self.user_to_review[userId] = []
                    self.user_to_review[userId].append(reviewId)

        print('Total books: %d\nTotal users: %d\nTotal reviews: %d\n'
              'Total book with review: %d\nTotal user with review: %d\n'
              % (len(self.book), len(self.user), len(self.review),
                 len(self.book_to_review), len(self.user_to_review)))
        # parse data into training and test.
        # since we read in review in order of books, the user in book_to_review are not in order.
        # we can leave out certain percentage of users in book_to_review to test.
        train_review = 0
        test_review = 0
        for bookId, reviewIdList in self.book_to_review.items():
            review_len = len(reviewIdList)
            if test_ratio * review_len > 0.8:
                start_idx = math.floor((1 - test_ratio) * review_len) + 1
                self.book_to_review_train[bookId] = reviewIdList[:start_idx]
                self.book_to_review_test[bookId] = reviewIdList[start_idx+1:]

                train_review += (start_idx + 1)
                test_review += (review_len - start_idx - 1)

                # construct user_to_review train and test

                for reviewId in reviewIdList[:start_idx]:
                    userId = self.review[reviewId][2]
                    if userId not in self.user_to_review_train:
                        self.user_to_review_train[userId] = []
                    self.user_to_review_train[userId].append(reviewId)

                for reviewId in reviewIdList[start_idx+1:]:
                    userId = self.review[reviewId][2]
                    if userId not in self.user_to_review_test:
                        self.user_to_review_test[userId] = []
                    self.user_to_review_test[userId].append(reviewId)
                #print('For book %d with %d reviews, 0 to %d are train and %d to %d are test.' %
                      #(bookId, review_len, start_idx-1, start_idx, review_len-1))

                #input('check result')
            else:
                train_review += review_len
                self.book_to_review_train[bookId] = reviewIdList
                for reviewId in reviewIdList:
                    userId = self.review[reviewId][2]
                    if userId not in self.user_to_review_train:
                        self.user_to_review_train[userId] = []
                    self.user_to_review_train[userId].append(reviewId)

        # check result
        print('Total user in train %d, in test %d, test ratio %f\n'
              'Total book in train %d, in test %d, test ratio %f\n'
              'Total review in train %d, in test %d, test ratio %f\n' %
              (len(self.user_to_review_train), len(self.user_to_review_test),
               len(self.user_to_review_test) / len(self.user),
               len(self.book_to_review_train), len(self.book_to_review_test),
               len(self.book_to_review_test) / len(self.book),
               train_review, test_review, test_review / len(self.review)))

    def get_full_data(self):
        return self.user, self.book, self.review, self.user_to_review, self.book_to_review

    def get_parsed_data(self):
        return self.book_to_review_train, self.book_to_review_test, \
               self.user_to_review_train, self.user_to_review_test

if __name__ == '__main__':
    db = Reader()
    user, book, review, user_to_review, book_to_review = db.get_full_data()