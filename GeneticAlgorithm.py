import numpy as np
from keras.models import Sequential
from typing import List

from TradingModelTrainer import TradingModelTrainer

class GeneticAlgorithm:
    def __init__(self, model_trainer: TradingModelTrainer):
        self.model_trainer = model_trainer
        self.best_model = None
        self.best_sharpe_ratio = -100000
        self.kek = False
    def genetic_algorithm(self, population: List[Sequential], generations: int, timesteps: int, features: int) -> List[Sequential]:
        for generation in range(generations):
            print("gen:", generation, "/", generations)
            traders = [self.model_trainer.simulate_trading(model, timesteps) for model in population]
            sharp_ratios = [trader.calculate_sharpe_ratio() for trader in traders]

            current_best_sharpe = max(sharp_ratios)
            current_best_model = population[np.argmax(sharp_ratios)]

            # Update the best model if the current generation's best is better
            if current_best_sharpe > self.best_sharpe_ratio:
                self.best_model = current_best_model
                self.best_sharpe_ratio = current_best_sharpe

            if all(sharpe_ratio == -100000 for sharpe_ratio in sharp_ratios):
                print("All models underperforming, generating new models.")
                population = [self.model_trainer.create_model(timesteps, features) for _ in range(len(population))]
                if self.best_model:
                    self.kek = True

            sorted_indices = np.argsort(sharp_ratios)[::-1]
            top_half_indices = sorted_indices[:len(population) // 10]
            top_half_models = [population[idx] for idx in top_half_indices]

            new_models = []
            while len(new_models) < len(population):
                parent_indices = np.random.choice(len(top_half_models), 2, replace=False)
                parent1 = top_half_models[parent_indices[0]]
                parent2 = top_half_models[parent_indices[1]] if not self.kek else self.best_model
                child = self.crossover_and_mutate(parent1, parent2, timesteps, features)
                new_models.append(child)

            population = new_models
            best_sharpe = max(sharp_ratios)
            print(f"Generation {generation + 1} complete. Best Sharpe Ratio: {best_sharpe}")
            self.kek = False
        return population
    
    def crossover_and_mutate(self, model1: Sequential, model2: Sequential, timesteps: int, features: int, mutation_rate=0.5, mutation_scale_range=(0.1, 0.5)) -> Sequential:
        """Combine and mutate two models to create a new model with random mutation scale."""
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        new_weights = []

        for w1, w2 in zip(weights1, weights2):
            mask = np.random.randint(0, 2, size=w1.shape)
            new_w = np.where(mask, w1, w2)
            if np.random.rand() < mutation_rate:
                # Randomly choose a mutation scale within the specified range
                random_mutation_scale = np.random.uniform(*mutation_scale_range)
                mutation = np.random.normal(loc=0.0, scale=abs(random_mutation_scale), size=new_w.shape)
                new_w += mutation
            new_weights.append(new_w)

        new_model = self.model_trainer.create_model(timesteps, features)
        new_model.set_weights(new_weights)
        return new_model

