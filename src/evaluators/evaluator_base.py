"""abstract base class for all evaluators"""
from sklearn.metrics import confusion_matrix, classification_report

class EvaluatorBase():
    """EvaluatorBase is abstract base class for all evaluators"""
    def get_evaluation_score_train_dev_test(self, tagger, datasets_bank, batch_size=-1):
        if batch_size == -1:
            batch_size = tagger.batch_size
        score_train, _, _ = self.predict_evaluation_score(tagger=tagger,
                                                       word_sequences=datasets_bank.word_sequences_train,
                                                       targets_tag_sequences=datasets_bank.tag_sequences_train,
                                                       batch_size=batch_size)
        score_dev, _, _= self.predict_evaluation_score(tagger=tagger,
                                                     word_sequences=datasets_bank.word_sequences_dev,
                                                     targets_tag_sequences=datasets_bank.tag_sequences_dev,
                                                     batch_size=batch_size)
        score_test, msg_test, clf_report = self.predict_evaluation_score(tagger=tagger,
                                                             word_sequences=datasets_bank.word_sequences_test,
                                                             targets_tag_sequences=datasets_bank.tag_sequences_test,
                                                             batch_size=batch_size)
        return score_train, score_dev, score_test, msg_test, clf_report

    def predict_evaluation_score(self, tagger, word_sequences, targets_tag_sequences, batch_size):
        outputs_tag_sequences = tagger.predict_tags_from_words(word_sequences, batch_size)
        acc, msg = self.get_evaluation_score(targets_tag_sequences, outputs_tag_sequences, word_sequences)
        msg += self.generate_classification_report(targets_tag_sequences, outputs_tag_sequences)
        #msg += self.generate_confusion_matrix(targets_tag_sequences, outputs_tag_sequences)
        return acc, msg, self.generate_classification_report(targets_tag_sequences, outputs_tag_sequences)

    def generate_confusion_matrix(self, truth, predicted, labels=None):
        ppredicted = sum(predicted, [])
        ttruth = sum(truth, [])
        if labels is None:
            labels = list(set(ttruth + ppredicted))
            return "\n" + confusion_matrix(ttruth, ppredicted, labels) + "\n"
        return "\n" + confusion_matrix(ttruth, ppredicted, labels) + "\n"

    def generate_classification_report(self, truth, predicted, labels=None):
        ppredicted = sum(predicted, [])
        ttruth = sum(truth, [])
        if labels is None:
            labels = list(set(ttruth + ppredicted))
            return "\n" + classification_report(ttruth, ppredicted, labels) + "\n"
        return "\n" + classification_report(ttruth, ppredicted, labels) + "\n"
