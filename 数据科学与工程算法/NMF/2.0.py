import argparse
import os
import random
from typing import Dict, List, NamedTuple, Tuple
import math

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--movie", default="./ml-10M100K/movies.dat")
parser.add_argument("--ratings", default="./ml-10M100K/ratings.dat")
parser.add_argument("--delimiter", default="::")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-feature", type=int, default=96)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lrate", type=int, default=0.001)
parser.add_argument("--decay", type=int, default=0.02)
parser.add_argument("--logging-interval", type=int, default=5000)


class Movie(NamedTuple):
    movie_id: int
    title: str
    genres: List[str]


class Rating(NamedTuple):
    user_id: int
    movie_id: int
    rating: float


def main():
    args = parser.parse_args()

    movies, ratings = read_dataset(args.movie, args.ratings, args.delimiter)

    random.seed(args.seed)
    np.random.seed(args.seed)

    random.shuffle(ratings)
    train_length = int(len(ratings) * 0.8)
    dev_length = int(len(ratings) * 0.1)

    ratings_train = ratings[:train_length]
    ratings_dev = ratings[train_length : train_length + dev_length]
    ratings_test = ratings[train_length + dev_length :]
    steps_per_epoch = math.ceil(len(ratings_train) / args.batch_size)

    user_length = max([rating.user_id for rating in ratings])
    movie_length = max(movies.keys())

    users_embedding = np.random.normal(0, 0.1, [user_length + 1, args.num_feature])
    movies_embedding = np.random.normal(0, 0.1, [movie_length + 1, args.num_feature])

    best_val_mae = 1e10
    running_loss = 0.0
    train_step_fn = train_step(args.lrate, args.decay)

    for epoch_number in range(args.epoch):
        for step_number in range(steps_per_epoch):
            batch = ratings_train[step_number * args.batch_size : (step_number + 1) * args.batch_size]
            user_id = tuple(rating.user_id for rating in batch)
            movie_id = tuple(rating.movie_id for rating in batch)
            y = np.array([rating.rating for rating in batch])

            users_embedding, movies_embedding, loss = train_step_fn(
                user_id,
                movie_id,
                y,
                users_embedding,
                movies_embedding,
            )

            running_loss += loss

            if step_number % args.logging_interval == args.logging_interval - 1:
                tqdm.write(
                    f"[Train] Epoch {epoch_number}, Step {step_number}, "
                    f"Loss: {running_loss / args.logging_interval}"
                )
                running_loss = 0.0

        # validate
        val_mae = validate(
            ratings_dev,
            users_embedding,
            movies_embedding,
        )

        tqdm.write(f"[Valid] Epoch {epoch_number}, Loss: {val_mae}")

        if best_val_mae > val_mae:
            best_val_mae = val_mae
        else:
            tqdm.write("Stop")
            break

    test_mae = validate(
        ratings_test,
        users_embedding,
        movies_embedding,
    )

    tqdm.write(f"[Test] Loss: {test_mae}")


def read_dataset(movie_filename, ratings_filename, delimiter):
    def parse_movie(line):
        movie_data = line.strip().split(delimiter)
        movie_id = int(movie_data[0])
        return movie_id, Movie(movie_id, movie_data[1], movie_data[2].split("|"))

    with open(movie_filename) as movie_file:
        movies = dict([parse_movie(line) for line in movie_file])

    def parse_rating(line):
        rating_data = line.strip().split(delimiter)
        return Rating(int(rating_data[0]), int(rating_data[1]), float(rating_data[2]))

    with open(ratings_filename) as ratings_file:
        ratings = [parse_rating(line) for line in ratings_file]

    return movies, ratings


def train_step(lrate, decay):
    def _inner(user_id, movie_id, y, users_embedding, movies_embedding):
        u = users_embedding[user_id, :]
        v = movies_embedding[movie_id, :]

        y_hat = np.sum(u * v, axis=1)
        r = np.expand_dims(y - y_hat, axis=1)

        users_embedding[user_id, :] += lrate * (r * v - decay * u)
        movies_embedding[movie_id, :] += lrate * (r * u - decay * v)

        return users_embedding, movies_embedding, np.sum(r)

    return _inner


def validate(ratings_dataset, users_embedding, movies_embedding):
    user_id = tuple(rating.user_id for rating in ratings_dataset)
    movie_id = tuple(rating.movie_id for rating in ratings_dataset)
    y = np.array([rating.rating for rating in ratings_dataset])

    u = users_embedding[user_id, :]
    v = movies_embedding[movie_id, :]

    y_hat = np.clip(np.sum(u * v, axis=1), 1.0, 5.0)
    mae = np.absolute(y - y_hat).mean()

    return mae


if __name__ == "__main__":
    main()
