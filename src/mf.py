from reader import Reader
import math
import random

import time

random_start = True
converge_change_rate = 10000

class Mf():
    def __init__(self, l2_reg = 0.1, l_rate = 0.1, dim = 20):
        self.l2_reg = l2_reg
        self.l_rate = l_rate
        self.dim = dim

        self.user_mat = {}
        self.book_mat = {}

        self.prev_avg_err = None

        self.unseen_user = 0
        self.unseen_book = 0

    def init_matrix(self, user_to_review, book_to_review):
        if not random_start:
            for id, _ in user_to_review.items():
                self.user_mat[id] = [0.5] * self.dim
            for id, _ in book_to_review.items():
                self.book_mat[id] = [0.5] * self.dim
        else:
            idx_print = 0
            print("Total users %d" % len(user_to_review.keys()))
            for id, _ in user_to_review.items():
                self.user_mat[id] = []
                for idx in range(self.dim):
                    self.user_mat[id].append(random.uniform(-1, 1))
                if idx_print % 10000 == 0:
                    print("Complete initializing %d user vectors." % idx_print, flush=True)
                idx_print+=1
            for id, _ in book_to_review.items():
                self.book_mat[id] = []
                for idx in range(self.dim):
                    self.book_mat[id].append(random.uniform(-1, 1))

    def get_predict(self, user_id, book_id):
        if user_id not in self.user_mat:
            self.unseen_user += 1
            #print('user id %d not seen in training set' % user_id)
            return None
        if book_id not in self.book_mat:
            self.unseen_book += 1
            #print('book id %d not seen in training set' % book_id)
            return None

        user_vec = self.user_mat[user_id]
        book_vec = self.book_mat[book_id]
        predict = 0
        for idx in range(self.dim):
            predict += user_vec[idx] * book_vec[idx]
        return predict

    def get_predictions(self, user_to_review, book_to_review, review):
        self.unseen_user = 0
        self.unseen_book = 0
        prediction = {}
        for book_id, book_review_arr in book_to_review.items():
            book_review_set = set(book_review_arr)
            # not iterating all users here since each book may only be rated by a few users.
            for review_id in book_review_set:
                user_id = review[review_id][2]
                # check if the user exists in training set.
                if user_id in user_to_review:
                    if user_id not in prediction:
                        prediction[user_id] = {}
                    prediction[user_id][book_id] = self.get_predict(user_id, book_id)
        print('Unseen user %d' % self.unseen_user)
        print('Unseen book %d' % self.unseen_book)
        return prediction

    def get_norm(self, vec):
        norm = 0
        for i in vec:
            norm += math.pow(i, 2)
        return math.sqrt(norm)

    def get_error(self, user_id, book_id, review_id, prediction, review):
        true_rating = review[review_id][0]
        pred_rating = prediction
        if pred_rating is None:
            # missing data. eg. New user, new book etc.
            # TODO: either fix dataset, or return some baseline for error(take avg as prediction). Currently returning 0
            return None
        return true_rating - pred_rating

    def get_errors(self, prediction, user_to_review, book_to_review, review):
        err_real = err = total_num = 0
        self.err_mat = {}
        for book_id, book_review_arr in book_to_review.items():
            review_set = set(book_review_arr)
            for review_id in review_set:
                user_id = review[review_id][2]
                if user_id not in self.err_mat:
                    self.err_mat[user_id] = {}
                self.err_mat[user_id][book_id] = self.get_error(user_id, book_id, review_id, prediction[user_id][book_id], review)
                # check for unseen
                if self.err_mat[user_id][book_id] is None:
                    # TODO: deal with None in dataset
                    continue
                # regularized error
                err += math.pow(self.err_mat[user_id][book_id], 2)
                err += self.l2_reg * (self.get_norm(self.user_mat[user_id]) + self.get_norm(self.book_mat[book_id]))
                err_real += math.fabs(self.err_mat[user_id][book_id])
                total_num += 1
        print('Get errors for %d items.' % total_num)
        return err, err_real/total_num

    def update_matrix(self):
        first = True
        firstbook = True
        for user_id, book_id_dict in self.err_mat.items():
            for book_id, err in book_id_dict.items():
                if firstbook:
                    print('user id %d book id %d err %f ' % (user_id, book_id, err))
                    firstbook = False
                for idx in range(self.dim):
                    prev_user_val = self.user_mat[user_id][idx]
                    prev_book_val = self.book_mat[book_id][idx]
                    self.user_mat[user_id][idx] += self.l_rate * (err * prev_book_val - self.l2_reg * prev_user_val)
                    # print('user %d val %d prev %f now %f'
                    # % (user_id, idx, prev_user_val, self.user_mat[user_id][idx]))
                    # input('pause')
                    self.book_mat[book_id][idx] += self.l_rate * (err * prev_user_val - self.l2_reg * prev_book_val)
            if first:
                print('user id %d' % user_id)
                print(self.user_mat[user_id])
                first = False

    def check_converged(self, prediction, user_to_review, book_to_review, review):
        err_avg, err_real_avg = self.get_errors(prediction, user_to_review, book_to_review, review)
        if self.prev_avg_err is not None and math.fabs(err_avg - self.prev_avg_err) < converge_change_rate:
            return True, err_avg, err_real_avg
        self.prev_avg_err = err_avg
        return False, err_avg, err_real_avg

    def train_gd(self, user_to_review, book_to_review, review):
        prediction = self.get_predictions(user_to_review, book_to_review, review)
        iter_count = 0
        while True:
            succ, err, err_real = self.check_converged(prediction, user_to_review, book_to_review, review)
            if succ:
                print('Train complete after %d iterations with train err %f.' % (iter_count, err))
                return
            if iter_count % 1 == 0:
                print('Train after %d iterations. Error: %f' % (iter_count, err))
                input('Pause')
            self.update_matrix()
            prediction = self.get_predictions(user_to_review, book_to_review, review)
            iter_count += 1

    def train_sgd(self, user_to_review, book_to_review, review, user_to_review_test, book_to_review_test):
        iter_count = 0
        loop_time_avg = 0
        review_keys = list(review.keys())

        # For generating test data only. Output starting error rate.
        prediction = self.get_predictions(user_to_review, book_to_review, review)
        succ, err_train, err_real = self.check_converged(prediction, user_to_review, book_to_review, review)
        print("Train Error: %f. Real average error: %f" % (err_train, err_real), flush=True)

        predictions = self.get_predictions(user_to_review_test, book_to_review_test, review)
        err_test, err_real = self.get_errors(predictions, user_to_review_test, book_to_review_test, review)
        print("Test Error: %f. Real average error: %f" % (err_test, err_real), flush=True)

        while True:
            # random choose one review
            #start_time = time.time()
            while True:
                review_id = random.choice(review_keys)
                book_id = review[review_id][1]
                user_id = review[review_id][2]
                if user_id in self.user_mat and book_id in self.book_mat:
                    break
                """
                if user_id in user_to_review and book_id in book_to_review \
                        and review_id in user_to_review[user_id] and review_id in book_to_review[book_id]:
                    break
                """
                loop_time_avg += 1
            #print('Pick random used %f' % (time.time() - start_time))
            #start_time = time.time()

            predict = self.get_predict(user_id, book_id)
            #print('Predict used %f' % (time.time() - start_time))
            #start_time = time.time()

            err = self.get_error(user_id, book_id, review_id, predict, review)
            #print('Error used %f' % (time.time() - start_time))
            #start_time = time.time()

            for idx in range(self.dim):
                prev_user_val = self.user_mat[user_id][idx]
                prev_book_val = self.book_mat[book_id][idx]
                self.user_mat[user_id][idx] += self.l_rate * (err * prev_book_val - self.l2_reg * prev_user_val)
                # print('user %d val %d prev %f now %f'
                # % (user_id, idx, prev_user_val, self.user_mat[user_id][idx]))
                # input('pause')
                self.book_mat[book_id][idx] += self.l_rate * (err * prev_user_val - self.l2_reg * prev_book_val)
            #print('Compute update used %f' % (time.time() - start_time))
            #start_time = time.time()

            if iter_count % 1000000 == 0:
                #print(self.user_mat[user_id])
                predict_after = self.get_predict(user_id, book_id)
                err_after = self.get_error(user_id, book_id, review_id, predict_after, review)
                print("Iteration %d, error of user_id %d, book_id %d is %f. Error after update %f"
                      % (iter_count, user_id, book_id, err, err_after), flush=True)
                print("Average loop time to find the next review is %d" % (loop_time_avg / 100000))
                loop_time_avg = 0
            if iter_count % 5000000 == 0 and iter_count > 0:
                prediction = self.get_predictions(user_to_review, book_to_review, review)
                succ, err_train, err_real = self.check_converged(prediction, user_to_review, book_to_review, review)
                print("Train Error: %f. Real average error: %f" % (err_train, err_real), flush=True)

                predictions = self.get_predictions(user_to_review_test, book_to_review_test, review)
                err_test, err_real = self.get_errors(predictions, user_to_review_test, book_to_review_test, review)
                print("Test Error: %f. Real average error: %f" % (err_test, err_real), flush=True)
                # Set different break out condition for benchmarking.
                if succ:
                    print("Train complete after %d iterations." % iter_count)
                    break
            iter_count += 1

    def matrix_factorization(self, user_to_review_train, book_to_review_train,
                             review, user_to_review_test, book_to_review_test):
        print('Initializing user, book matrices.')
        self.init_matrix(user_to_review_train, book_to_review_train)
        print('Start training.')
        self.train_sgd(user_to_review_train, book_to_review_train, review, user_to_review_test, book_to_review_test)
        predictions = self.get_predictions(user_to_review_test, book_to_review_test, review)
        return self.get_errors(predictions, user_to_review_test, book_to_review_test, review)

def cv(user_to_review_train, book_to_review_train, review, user_to_review_test, book_to_review_test):
    l2_reg_arr = [0.1, 0.01]
    l_rate_arr = [0.02, 0.00001]
    dim_arr = [100, 1000]

    for l2_reg in l2_reg_arr:
        for l_rate in l_rate_arr:
            for dim in dim_arr:
                mf = Mf(l2_reg, l_rate, dim)
                print("Testing for learning rate %f, l2 regularization %f and dimension %d."
                      % (l_rate, l2_reg, dim))
                err_avg, err_real_avg = mf.matrix_factorization(user_to_review_train, book_to_review_train,
                                        review, user_to_review_test, book_to_review_test)
                print("Average MSE: %f. Average error: %f" % (err_avg, err_real_avg))
                input('Pause')

if __name__ == '__main__':
    db = Reader()
    user, book, review, user_to_review, book_to_review = db.get_full_data()
    book_to_review_train, book_to_review_test, user_to_review_train, user_to_review_test = db.get_parsed_data()

    # test prediction
    cv(user_to_review_train, book_to_review_train, review, user_to_review_test, book_to_review_test)
