from typing import Literal

import numpy as np
from scipy.stats import entropy, ks_2samp, skewnorm


class FeatureGenerator:
    """
    Parent class for further metrics.
    Generates synthetic feature distributions for train and current data
    """

    def __init__(
        self,
        train_mean: float,
        train_std: float,
        train_size: int,
        current_mean: float,
        current_std: float,
        current_size: int,
        distribution: Literal["normal", "skewed"],
        **kwargs,
    ):
        self.train_mean = train_mean
        self.train_std = train_std
        self.train_size = train_size

        self.current_mean = current_mean
        self.current_std = current_std
        self.current_size = current_size

        self.distribution = distribution

        self.train = self.feature_generator(
            mean_=train_mean, std_=train_std, size=train_size, distribution="normal"
        )
        self.current = self.feature_generator(
            mean_=current_mean,
            std_=current_std,
            size=current_size,
            distribution=distribution,
            **kwargs,
        )

    @staticmethod
    def feature_generator(
        mean_: float,
        std_: float,
        size: int,
        distribution: Literal["normal", "skewed"] = "normal",
        **kwargs,
    ) -> np.ndarray:
        """
        Generates a synthetic feature distribution.
        """
        np.random.seed(0)
        if distribution == "normal":
            return np.random.normal(loc=mean_, scale=std_, size=size)
        elif distribution == "skewed":
            skew = kwargs.get("skew", 4)
            return np.array(skewnorm.rvs(a=skew, loc=mean_, scale=std_, size=size))
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")


class FeatureGeneratorWhithBins(FeatureGenerator):
    def __init__(self, n_batches: int, **kwargs):
        super().__init__(**kwargs)
        self.n_batches = n_batches
        self.bins = np.linspace(
            start=min(self.train), stop=max(self.train), num=self.n_batches + 1
        )
        self.train_ratio = self.calculate_ratio(self.train, self.bins)
        self.current_ratio = self.calculate_ratio(self.current, self.bins)

    def calculate_ratio(self, data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Вычисляет долю наблюдений в каждом бине с защитой от нулевых значений.
        Это нормированные гистограммы: cумма значений по бинам равна 1

        Параметры:
        ----------
        - data: Одномерный массив данных, для которого нужно рассчитать доли вхождений по бинам.
        - bins: массив бинов (интервалов) для разбиения диапазона значений.

        Возвращает:
        ----------
        - массив нормированных долей по каждому бину, в котором нули заменены на 1e-6 (защита от логарифма нуля).
        """
        counts, _ = np.histogram(data, bins=bins, density=False)
        train_ratio = counts / len(data)
        zero_protected = np.clip(train_ratio, 1e-6, None)
        return zero_protected


class PSI(FeatureGeneratorWhithBins):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = self.calculate_psi(self.train_ratio, self.current_ratio)

    def __call__(self) -> float:
        return self.value

    @staticmethod
    def calculate_psi(train_ratio, current_ratio) -> float:
        """
        Векторный расчет PSI

        Параметры:
        ---------
        - train_ration: массив нормированных долей по каждому бину в ОБУЧАЮЩЕЙ ВЫБОРКЕ. Число элементов в массиве определено числом бинов
        - current_ratio: массив нормированных долей по каждому бину в ТЕКУЩЕЙ ВЫБОРКЕ. Число элементов в массиве определено числом бинов
        """
        psi_values = (train_ratio - current_ratio) * np.log(train_ratio / current_ratio)
        return np.sum(psi_values)


class Wasserstein(FeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.common_x = self.make_common_space()

        # Квантильная Функция (обратная эмпирическая функция распределения) -> IЕCDF
        self.train_iecdf = self.make_iecdf(self.train, self.common_x)
        self.current_iecdf = self.make_iecdf(self.current, self.common_x)
        self.value = self.wasserstein()
        self.value_norm = self.wasserstein_norm()

    def __call__(self) -> float:
        return self.value_norm

    def make_common_space(self):
        """
        Возвращает массив равномерно распределенных точек в диапазоне от 0 до 1.
        Число элементов в массиве соответствует наибольшей выборке (из двух)
        """
        return np.linspace(0, 1, max(self.current_size, self.train_size))

    @staticmethod
    def make_iecdf(data: np.ndarray, common_x: np.ndarray):
        """
        Возвращает значения квантильной функции (inverse ECDF (empirical cumulative distribution function)).
        - В условиях нашей задачи это нужно, чтобы "достроить" недостающие значения обеим выборкам так, чтобы они легли на одну OX и полностью ее заполнили.
        - Это даст 2 IECDF, по которым можно будет посчитать интеграл разницы.

        Логика:
        - np.interp: функция интерполяции. Принимает:
            - x: массив значений аргумента, ДЛЯ КОТОРЫХ нужно вернуть значения функции
            - xp: массив значений аргумента, для которых ИЗВЕСТНЫ значения функции
            - fp: массив ИЗВЕСТНЫХ значений функции
        """
        return np.interp(
            x=common_x,
            xp=np.linspace(0, 1, len(data), endpoint=False),
            fp=np.sort(data),
        )

    def wasserstein(self):
        """
        Благодаря тому, что на предыдущем шаге рассчитаны обратные CDF, теперь есть 2 массива квантилей от 0 до 1 с соответствующими значениями.
        Т.к. площадь считается на всем интервале вероятностей (от 0 до 1), то интегрирование сводится к рассчету среднего значения по OY,
            что в приближении равно просто среднему значению разницы квантильных функций двух распределений.
        """
        return np.mean(np.abs(self.train_iecdf - self.current_iecdf))

    def wasserstein_norm(self):
        """
        Нормализует расстояние по диапазону значений фичи
        """
        scale = max(self.train.max(), self.current.max()) - min(
            self.train.min(), self.current.min()
        )
        return self.value / scale if scale else 0.0


class KS(FeatureGenerator):
    def __init__(self, **kwargs):
        """
        all_uniques: Объединённая ось X (все уникальные значения из обеих выборок)
        ks_statistic: the maximum absolute difference between ECDFs
        """
        super().__init__(**kwargs)
        self.all_uniques = np.sort(
            np.unique(np.concatenate([self.train, self.current]))
        )
        self.train_ecfd = self.make_ecdf(self.train)
        self.current_ecfd = self.make_ecdf(self.current)
        self.ks_statistic = np.max(np.abs(self.train_ecfd - self.current_ecfd))
        self.p_value = self.ks_p_value()

    def __call__(self) -> float:
        return self.ks_statistic

    def ks_p_value(self) -> np.float64:
        """
        e p-value for the test
        """
        _, pval = ks_2samp(self.train, self.current)
        return pval  # type: ignore

    def make_ecdf(self, data: np.ndarray) -> np.ndarray:
        """
        Returns ECDF for a given array.
        Logic:
        -----
        - np.searchsorted: возвращает позиции передаваемых элементов, на которые их нужно поставить в отсортированном массиве.
            При нормализации дает ECDF
                -a: отсортированный массив
                -v: значения, для которых нужно найти места
                -side: ['left', 'right'] - с какой стороны массы дубликатов ставим. Нам нужен "right"
        """
        sorted_data = np.sort(data)

        return np.searchsorted(a=sorted_data, v=self.all_uniques, side="right") / len(
            sorted_data
        )


class JansenShannon(FeatureGeneratorWhithBins):
    def __init__(self, **kwargs):
        """ 
        self.M - "среднее" распределение:
                - у нас уже есть нормированные гистограммы, в которых по каждому бину дается доля вхождений.
                - Эти значения усредняются для двух гистограмм (т.е. половина поэлементной суммы)
                - Расстояния от этого "среднего" распределения и будут учитываться в метрике 
        """
        super().__init__(**kwargs)
        self.M = 0.5 * (self.train_ratio + self.current_ratio)
        self.value = self.compute_js_distance()
        self.jsd_contrib = self.compute_bin_contributions()

    def __call__(self):
        return self.value

    def compute_js_distance(self) -> float:
        """
        Compute Jensen-Shannon distance between train and current distributions.
        1. Расчитываем "среднее" распределение:
            - у нас уже есть нормированные гистограммы, в которых по каждому бину дается доля вхождений.
            - Эти значения усредняются для двух гистограмм (т.е. половина поэлементной суммы)
            - Расстояния от этого "среднего" распределения и будут учитываться в метрике

        2. Расчитываем Kullback–Leibler divergence между каждым из распределений и их "средним".
            >>> stats.entropy (p) -> рассчитывает Shannon Entropy
            >>> stats.entropy (p, q) -> рассчитывает Kullback–Leibler divergence (related entropy)
            >>> справочно: KLD по природе - это разница кросс-энтропии и энтропии
                (которая всегда больше нуля и равна нулю только в случае совпадения распределений).
                KLD = H(p, q) - H(p)
                KLD = -sum(p * log(q)) + sum(p * log(p))
                KLD = sum (p * log(p/q))

        3. Вычисляем Jensen–Shannon divergence как среднее из двух KLD:
            >>> JSD = 0.5 * (KL(train || M) + KL(current || M))

        4. Преобразуем дивергенцию в Jensen–Shannon distance:
            >>> Метрика лежит в диапазоне [0, 1] и симметрична
            >>> Интуиция: при расчете KLD мы брали двоичные логарифмы от вероятностей [0,1], т.е. возводили во вторую степень
                Извлечение квадрата возвращает значения в диапазон [0,1]
        """

        kld_train = entropy(self.train_ratio, self.M, base=2)
        kld_current = entropy(self.current_ratio, self.M, base=2)

        js_divergence = 0.5 * (kld_train + kld_current)
        js_distance = np.sqrt(js_divergence)
        return float(js_distance)

    def compute_bin_contributions(self)-> np.ndarray:
        """
        Compute per-bin contributions to the Jensen–Shannon divergence.
        Возвращает:
        ----------
        - Массив длины n_bins: вклад каждого бина в JSD
            >>> JSD по бинам = 0.5 * (p * log2(p / M) + q * log2(q / M))
        """
        kl_pm = self.train_ratio * np.log2(self.train_ratio / self.M)
        kl_qm = self.current_ratio * np.log2(self.current_ratio / self.M)
        return 0.5 * (kl_pm + kl_qm)

