from typing import List

from tensorflow_core.python.keras import Sequential, Input, Model
from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, concatenate

from aiton_senna.ai import AI
from fsai.car.car import Car
from fsai.objects.track import Track


class EvolutionarySimulation:
    def __init__(self):
        self.track = None
        self.blue_boundary, self.yellow_boundary, self.o, self.all_boundaries = [], [], [], []

        self.base_car = None
        self.furthest_distance = 0
        self.base_model = self.gen_model(new=True)

        self.episode_running = False
        self.episode_count = 0
        self.episode_time = 0

        self.step_size = 0.2

        self.ai = []
        self.best_ai = None

    def new_episode(self, car_count: int = 10):
        self.episode_running = True
        self.episode_time = 0

        for i in range(car_count):
            self.ai.append(AI(self))

    def update(self, dt: float):
        for ai in self.get_alive_ai():
            ai.update(dt)

        if len(self.get_alive_ai()) == 0:
            self.on_episode_end()

    def on_episode_end(self):
        for ai in self.ai:
            if ai.distance > self.furthest_distance:
                self.base_model = ai.model
                self.furthest_distance = ai.distance
        print("Episode: {} distance {}".format(self.episode_count, self.furthest_distance))

        self.episode_count += 1
        self.episode_running = False

    def get_alive_ai(self) -> List[AI]:
        return [ai for ai in self.ai if ai.alive]

    def set_track(self, track: Track):
        self.track = track
        self.blue_boundary, self.yellow_boundary, self.o = track.get_boundary()
        self.all_boundaries = self.blue_boundary + self.yellow_boundary + self.o
        self.base_car = track.cars[0]

    def gen_model(self, new=False):
        # define two sets of inputs
        inputA = Input(shape=(120, 80, 3))
        inputB = Input(shape=7)

        # the first branch operates on the first input
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1')(inputA)
        x = MaxPooling2D((2, 2), name='maxpool_1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
        x = MaxPooling2D((2, 2), name='maxpool_2')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3')(x)
        x = MaxPooling2D((2, 2), name='maxpool_3')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_4')(x)
        x = MaxPooling2D((2, 2), name='maxpool_4')(x)
        x = Flatten()(x)
        x = Model(inputs=inputA, outputs=x)

        # the second branch opreates on the second input
        y = Dense(8, activation="sigmoid")(inputB)
        y = Dense(4, activation="sigmoid")(y)
        y = Model(inputs=inputB, outputs=y)

        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(30, activation="sigmoid")(combined)
        z = Dense(20, activation="sigmoid")(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)

        if not new:
            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i] * self.step_size + self.base_model.weights[i] * (1 - self.step_size)
        return model
