import os
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE

from catboost import CatBoostClassifier

from utils import RANDOM_STATE


os.makedirs("outputs", exist_ok=True)

data = joblib.load("outputs/preprocessed_data.pkl")

X = data["X"]
y = data["y"]

rng = np.random.default_rng(RANDOM_STATE)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


def sample_parameter(space):
    kind = space[0]

    if kind == "int":
        return int(rng.integers(space[1], space[2] + 1))

    if kind == "float":
        return float(rng.uniform(space[1], space[2]))

    if kind == "cat":
        return rng.choice(space[1]).item()

    raise ValueError("Unknown parameter type")


def create_individual(search_space):
    return {
        param: sample_parameter(space)
        for param, space in search_space.items()
    }


def mutate(individual, search_space, mutation_rate=0.2):
    mutated = individual.copy()

    for param in mutated:
        if rng.random() < mutation_rate:
            mutated[param] = sample_parameter(search_space[param])

    return mutated


def crossover(parent1, parent2):
    child = {}

    for param in parent1:
        child[param] = parent1[param] if rng.random() < 0.5 else parent2[param]

    return child


def evaluate_individual(model, params, X, y):
    candidate = clone(model)
    candidate.set_params(**params)

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", BorderlineSMOTE(random_state=RANDOM_STATE)),
        ("model", candidate)
    ])

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    return scores.mean()


def evolutionary_optimization(
    model,
    search_space,
    X,
    y,
    population_size=20,
    generations=100,
    mutation_rate=0.2,
    elite_fraction=0.25
):
    population = [
        create_individual(search_space)
        for _ in range(population_size)
    ]

    history = []

    best_score = -np.inf
    best_params = None

    n_elites = max(2, int(population_size * elite_fraction))

    for generation in range(1, generations + 1):

        scored_population = []

        for individual in population:
            score = evaluate_individual(
                model=model,
                params=individual,
                X=X,
                y=y
            )

            scored_population.append((score, individual))

            if score > best_score:
                best_score = score
                best_params = individual.copy()

        scored_population.sort(
            key=lambda x: x[0],
            reverse=True
        )

        elites = [
            individual
            for score, individual in scored_population[:n_elites]
        ]

        history.append({
            "Generation": generation,
            "Best_Generation_Score": scored_population[0][0],
            "Best_Overall_Score": best_score,
            "Best_Generation_Params": scored_population[0][1],
            "Best_Overall_Params": best_params.copy()
        })

        print(
            f"Generation {generation:03d} | "
            f"Generation best: {scored_population[0][0]:.4f} | "
            f"Overall best: {best_score:.4f}"
        )

        next_population = elites.copy()

        while len(next_population) < population_size:
            p1_idx, p2_idx = rng.integers(0, len(elites), size=2)

            child = crossover(
                elites[p1_idx],
                elites[p2_idx]
            )

            child = mutate(
                child,
                search_space,
                mutation_rate=mutation_rate
            )

            next_population.append(child)

        population = next_population

    return best_params, best_score, pd.DataFrame(history)


M1_search_space = {
    "bootstrap": ("cat", [True, False]),
    "max_samples": ("int", 1, 20),
    "n_estimators": ("int", 1, 200),
    "max_features": ("int", 1, 4)
}

M3_search_space = {
    "n_estimators": ("int", 10, 200),
    "max_depth": ("int", 5, 50),
    "max_leaf_nodes": ("int", 5, 50),
    "min_samples_split": ("int", 2, 20),
    "min_samples_leaf": ("int", 1, 10),
    "criterion": ("cat", ["gini", "entropy"])
}

M7_search_space = {
    "depth": ("int", 3, 10),
    "learning_rate": ("float", 0.01, 0.7),
    "iterations": ("int", 50, 300)
}


M1_base = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    random_state=RANDOM_STATE
)

M3_base = ExtraTreesClassifier(
    bootstrap=True,
    random_state=RANDOM_STATE
)

M7_base = CatBoostClassifier(
    loss_function="MultiClass",
    verbose=0,
    random_seed=RANDOM_STATE
)


for model_name, base_model, search_space in [
    ("M1", M1_base, M1_search_space),
    ("M3", M3_base, M3_search_space),
    ("M7", M7_base, M7_search_space)
]:

    print(f"\nRunning evolutionary optimization for {model_name}")

    best_params, best_score, history = evolutionary_optimization(
        model=base_model,
        search_space=search_space,
        X=X,
        y=y,
        population_size=20,
        generations=100,
        mutation_rate=0.2
    )

    history.to_csv(
        f"outputs/{model_name}_evolutionary_search_history.csv",
        index=False
    )

    joblib.dump(
        {
            "best_params": best_params,
            "best_score": best_score
        },
        f"outputs/{model_name}_best_evolutionary_params.pkl"
    )

    print(f"Best {model_name} params:", best_params)
    print(f"Best {model_name} score:", best_score)
