# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:37:19 2023

@author: Tiago
"""
import tensorflow as tf
class Precision(tf.keras.metrics.Metric):
    def __init__(self, num_classes, **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.precision = self.add_weight(name='precision', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.true_positives = [self.add_weight(name='true_positives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.false_positives = [self.add_weight(name='false_positives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        for i in range(self.num_classes):
            true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            self.true_positives[i].assign_add(true_positives)
            self.false_positives[i].assign_add(false_positives)
            denominator = self.true_positives[i] + self.false_positives[i]
            precision = tf.where(tf.equal(denominator, 0), 0.0, self.true_positives[i] / denominator)
            self.precision[i].assign(precision)

    def result(self):
        macro_precision = tf.reduce_mean(self.precision)
        return macro_precision

    def reset_state(self):
        self.precision.assign(tf.zeros(shape=(self.num_classes,), dtype=tf.float32))
        for i in range(self.num_classes):
            self.true_positives[i].assign(tf.zeros(shape=(), dtype=tf.float32))
            self.false_positives[i].assign(tf.zeros(shape=(), dtype=tf.float32))

class Recall(tf.keras.metrics.Metric):
    def __init__(self, num_classes, **kwargs):
        super(Recall, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.recall = self.add_weight(name='recall', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.true_positives = [self.add_weight(name='true_positives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.false_negatives = [self.add_weight(name='false_negatives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        for i in range(self.num_classes):
            true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i)), tf.float32))
            false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, i), tf.not_equal(y_pred, i)), tf.float32))
            self.true_positives[i].assign_add(tf.cast(true_positives, tf.float32))
            self.false_negatives[i].assign_add(tf.cast(false_negatives, tf.float32))
            denominator = self.true_positives[i] + self.false_negatives[i]
            recall = tf.where(tf.equal(denominator, 0), 0.0, self.true_positives[i] / denominator)
            self.recall[i].assign(recall)

    def result(self):
        macro_recall = tf.reduce_mean(self.recall)
        return macro_recall

    def reset_state(self):
        self.recall.assign(tf.zeros(shape=(self.num_classes,), dtype=tf.float32))
        for i in range(self.num_classes):
            self.true_positives[i].assign(tf.zeros(shape=(), dtype=tf.float32))
            self.false_negatives[i].assign(tf.zeros(shape=(), dtype=tf.float32))


class WeightedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, class_weights, **kwargs):
        super(WeightedAccuracy, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.weighted_accuracy = self.add_weight(name='weighted_accuracy', shape=(), initializer='zeros', dtype=tf.float32)
        self.total_samples = self.add_weight(name='total_samples', shape=(), initializer='zeros', dtype=tf.float32)
        self.true_positives = [self.add_weight(name='true_positives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.false_positives = [self.add_weight(name='false_positives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
        self.false_negatives = [self.add_weight(name='false_negatives_{}'.format(i), shape=(), initializer='zeros', dtype=tf.float32) for i in range(num_classes)]
    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        else:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
    
        for i in range(self.num_classes):
            y_true_i = tf.equal(y_true, i)
            y_pred_i = tf.equal(y_pred, i)
            true_positives = tf.math.count_nonzero(tf.logical_and(y_true_i, y_pred_i), dtype=tf.float32) * sample_weight * self.class_weights[i]
            false_positives = tf.math.count_nonzero(tf.logical_and(tf.logical_not(y_true_i), y_pred_i), dtype=tf.float32) * sample_weight * self.class_weights[i]
            false_negatives = tf.math.count_nonzero(tf.logical_and(y_true_i, tf.logical_not(y_pred_i)), dtype=tf.float32) * sample_weight * self.class_weights[i]
            self.true_positives[i].assign_add(tf.reduce_sum(true_positives))
            self.false_positives[i].assign_add(tf.reduce_sum(false_positives))
            self.false_negatives[i].assign_add(tf.reduce_sum(false_negatives))
        
        self.total_samples.assign_add(tf.reduce_sum(sample_weight))


    def result(self):
        if self.class_weights is None:
            return 0
        weighted_accuracies = [tp / (tp + fn + 1e-10) for tp, fn in zip(self.true_positives, self.false_negatives)]
        self.weighted_accuracy.assign(tf.reduce_sum([wa * cw for wa, cw in zip(weighted_accuracies, self.class_weights)]) / tf.reduce_sum(self.class_weights))
        return self.weighted_accuracy