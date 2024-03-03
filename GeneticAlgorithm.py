import numpy as np
from keras.models import Sequential
from typing import List

from TradingModelTrainer import TradingModelTrainer

class GeneticAlgorithm:
    def __init__(self, model_trainer: TradingModelTrainer):
        self.model_trainer = model_trainer

    def genetic_algorithm(self, population: List[Sequential], generations: int, timesteps: int, features: int) -> List[Sequential]:
        """Run a genetic algorithm for evolving models."""
        for generation in range(generations):
            print("gen:", generation, "/", generations)
            # Evaluate models using the simulate_trading method
            traders = [self.model_trainer.simulate_trading(model, timesteps) for model in population]
            sharp_ratios = [ i.calculate_sharpe_ratio() for i in traders ]
            top_models = []

            sorted_indices = np.argsort(sharp_ratios)[::-1]
            for i in sorted_indices[:len(population) // 2]:
                top_models.append(population[i])
            best_performed = traders[sorted_indices[0]]
            self.model_trainer.update_learning_history(best_performed.current_capital, best_performed.calculate_sharpe_ratio())

            new_models = []
            for _ in range(len(population) - len(top_models)):
                parents = np.random.choice(top_models, 2, replace=False)
                child = self.crossover_and_mutate(parents[0], parents[1], timesteps, features)
                new_models.append(child)
            population = top_models + new_models
            print(f"Generation {generation + 1} complete. Best Sharpe Ratio: {sharp_ratios[sorted_indices[0]]}")

        return population
    
    def crossover_and_mutate(self, model1: Sequential, model2: Sequential, timesteps: int, features: int, mutation_rate=0.3, mutation_scale=0.3) -> Sequential:
        """Combine and mutate two models to create a new model."""
        weights1 = model1.get_weights()
        weights2 = model2.get_weights()
        new_weights = []

        for w1, w2 in zip(weights1, weights2):
            mask = np.random.randint(0, 2, size=w1.shape)
            new_w = np.where(mask, w1, w2)
            if np.random.rand() < mutation_rate:
                mutation = np.random.normal(loc=0.0, scale=mutation_scale, size=new_w.shape)
                new_w += mutation
            new_weights.append(new_w)

        new_model = self.model_trainer.create_lstm_model(timesteps, features)
        new_model.set_weights(new_weights)
        return new_model