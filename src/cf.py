from reader import Reader
import math

import random
from scipy.stats.stats import pearsonr

def corr_sim(i, j, item_to_review, review):
    # find user who both rated the two items.
    def get_user_id(idx):
        return set([review[reviewId][2] for reviewId in item_to_review[idx]])

    user_id_i = get_user_id(i)
    user_id_j = get_user_id(j)
    # might have duplicated ratings.
    user_id_common = set([id for id in user_id_i if id in user_id_j])

    if len(user_id_common) == 0:
        return 0

    def get_user_rate(i):
        raw_rating = sorted([(review[reviewId][0], review[reviewId][2], reviewId)
                   for reviewId in item_to_review[i] if review[reviewId][2] in user_id_common], key=lambda a: a[1])
        # merge duplicate rating from same user
        agg_rating = {}
        res_rating = []
        for rate, user, _ in raw_rating:
            if user not in agg_rating:
                agg_rating[user] = []
            agg_rating[user].append(rate)
        for user, rate_arr in agg_rating.items():
            res_rating.append((sum(rate_arr) / len(rate_arr), user))
        return sorted(res_rating, key=lambda a:a[1])

    user_rate_i = get_user_rate(i)
    user_rate_j = get_user_rate(j)

    """
    print('item %d and %d' % (i, j))
    print(user_rate_i)
    print(user_rate_j)
    """

    def get_rate_avg(rate_arr):
        return sum([a[0] for a in rate_arr]) / len(rate_arr)

    avg_rate_i = get_rate_avg(user_rate_i)
    avg_rate_j = get_rate_avg(user_rate_j)
    try:
        nomi= sum([(user_rate_i[idx][0] - avg_rate_i) * (user_rate_j[idx][0] - avg_rate_j)
                   for idx in range(len(user_id_common))])
        denomi_i = math.sqrt(sum([math.pow((user_rate_i[idx][0] - avg_rate_i), 2)
                                  for idx in range(len(user_id_common))]))
        denomi_j = math.sqrt(sum([math.pow((user_rate_j[idx][0] - avg_rate_j), 2)
                                  for idx in range(len(user_id_common))]))
    except:
        print(user_id_i)
        print(user_id_j)
        print(user_rate_i)
        print(user_rate_j)
        print(user_id_common)
        input('error: pause')

    if (nomi == 0) and (denomi_i * denomi_j == 0):
        return 0

    """
    print(nomi)
    print(denomi_i)
    print(denomi_j)
    print(nomi / (denomi_i * denomi_j))
    print('pearson %f' % pearsonr([user_rate_i[idx][0] for idx in range(len(user_id_common))]
                                  , [user_rate_j[idx][0] for idx in range(len(user_id_common))])[0])
    input('pause')
    """

    return nomi / (denomi_i * denomi_j)

def compute_corr_sim(book_to_review, review):
    sim = {}
    book_id = [id for id, review_arr in book_to_review.items()]
    print('Total book ids: %d' % len(book_id))
    try:
        sim_file = open('../data/corr_sim.txt', 'r')
        print('Read sim from file')
        idx = 0
        for line in sim_file:
            vals = line.strip().split(' ')
            #print('vals len %d' % len(vals))
            id = book_id[idx]
            if id not in sim:
                sim[id] = {}
            for idx_j, val in enumerate(vals):
                sim[id][book_id[idx_j]] = float(val)
            idx += 1
        sim_file.close()
    except:
        print('Sim file not exist, computing...')
        input('Pause')
        for i in book_id:
            if i % 100 == 0:
                print('Complete %d books. ' % i)
            for j in book_id:
                if j % 500 == 0:
                    print('Complete %d books with %d. ' % (j, i))
                if i == j:
                    continue
                if i not in sim:
                    sim[i] = {}
                sim[i][j] = corr_sim(i, j, book_to_review, review)
            #print(sim[i])
            #input('pause')

        sim_file = open('../data/corr_sim.txt', 'w')
        for i in book_id:
            for j in book_id:
                if i == j:
                    sim_file.write('0 ')
                else:
                    sim_file.write('%f ' % sim[i][j])
            sim_file.write('\n')
        sim_file.close()

    print('Finish similarity computation.')
    return sim

def compute_pred(user_id, item_id, user_to_review, book_to_review, review, sim):
    rating = 0
    if user_id not in user_to_review:
        #print('user_id %d not found in train.' % user_id)
        return rating, False

    user_review_arr = set(user_to_review[user_id])

    book_id = set([review[id][1] for id in user_review_arr])
    book_id_rated = set([(review[id][1], review[id][0]) for id in user_review_arr])

    if item_id in book_id:
        # already rated
        rating = [x[1] for x in book_id_rated if x[0] == item_id][0]
    # check how users rate other books
    for id, rate in book_id_rated:
        rating += sim[item_id][id] * rate
    deno = sum([math.fabs(sim[item_id][id]) for id in book_id])
    rating = (rating / deno) if deno != 0 else 0
    return rating, True

""" Item based CF
"""
def collab_filter(user_to_review, book_to_review, review, user_to_review_test, book_to_review_test):
    #compute sim
    sim = compute_corr_sim(book_to_review, review)

    err_rate = total_test = succ_test = fail_test = 0
    baseline_fix_err_rate = 0
    baseline_rand_err_rate = 0

    for user_id, user_review_arr in user_to_review_test.items():
        for review_id in user_review_arr:
            item_id = review[review_id][1]
            rating_true = review[review_id][0]

            rating_pred, succ = compute_pred(user_id, item_id, user_to_review, book_to_review, review, sim)

            total_test += 1
            if succ:
                succ_test += 1
                err_rate += math.fabs(rating_true - rating_pred)

                baseline_fix_err_rate += math.fabs(rating_true - 3)
                baseline_rand_err_rate += math.fabs(rating_true - random.uniform(0, 5))
            else:
                fail_test += 1

            #print('Rating of user %d item %d. True: %f, Predict: %f'
                  #% (user_id, item_id, rating_true, rating_pred))
            #input('pause')
    err_rate /= succ_test
    baseline_fix_err_rate /= succ_test
    baseline_rand_err_rate /= succ_test
    print('Total test %d, %d success and %d fail. Average err: %f'
          % (total_test, succ_test, fail_test, err_rate))
    print('Baseline fixed average rating error: %f\nBaseline random average error: %f'
          % (baseline_fix_err_rate, baseline_rand_err_rate))
    # fill up rating
    """
    rating = {}
    idx_i = idx_u = 0
    for user_id, user_review_arr in user_to_review.items():
        idx_i = 0
        for item_id, item_review_arr in book_to_review.items():
            book_id = set([review[id][1] for id in user_review_arr])
            book_id_rated = set([(review[id][1], review[id][0]) for id in user_review_arr])

            if user_id not in rating:
                rating[user_id] = {}
            if item_id in book_id:
                # already rated
                rating[user_id][item_id] = [x[1] for x in book_id_rated if x[0] == item_id][0]
                continue
            # check how users rate other books
            pred_rate = 0
            for id, rate in book_id_rated:
                pred_rate += sim[item_id][id] * rate
            deno = sum([math.fabs(sim[item_id][id]) for id in book_id])
            pred_rate = (pred_rate / deno) if deno != 0 else 0
            if pred_rate != 0:
                rating[user_id][item_id] = pred_rate

            #print('user %d item %d rating %f' % (user_id, item_id, pred_rate))
            #input('pause')

            idx_i += 1
        idx_u += 1
        if idx_u % 500 == 0:
            print("Complete user %d." % idx_u)
    """

if __name__ == '__main__':
    db = Reader()
    user, book, review, user_to_review, book_to_review = db.get_full_data()
    book_to_review_train, book_to_review_test, user_to_review_train, user_to_review_test = db.get_parsed_data()

    # test corr_sim
    collab_filter(user_to_review_train, book_to_review_train, review, user_to_review_test, book_to_review_test)