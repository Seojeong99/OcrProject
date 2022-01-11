import tensorflow as tf
import numpy, os, re, random, pickle
from skimage import io, transform
from sklearn.model_selection import KFold


class NNOCRModel:
    LEARNING_RATE = 0.001
    BATCH_SIZE = 100
    DISPLAY_STEP = 10

    # layer1과 layer2는 히든 레이어의 크기입니다.
    # imgSize는 입력 받을 이미지의 크기입니다
    def __init__(self, layer1, layer2, imgSize):
        self.nLayer1 = layer1
        self.nLayer2 = layer2
        self.nImgSize = imgSize
        self.nInput = imgSize ** 2
        self.nClasses = 0
        self.oNames = []
        self.xhat = None
        self.yhat = None
        self.sess = None

    def __del__(self):
        if self.sess: self.sess.close()

    @staticmethod
    def restoreName(s):
        return re.sub('0x([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    def loadImg(self, path):
        image = io.imread(path)
        image = numpy.invert(image)
        image = numpy.average(image, axis=2)
        xs = transform.resize(image, (self.nImgSize, self.nImgSize))
        xs = numpy.reshape(xs, self.nInput).astype(float)
        xs /= numpy.max(xs)
        return xs

    # datasetFolder 내부로부터 이미지를 읽어들입니다.
    # datasetFolder 안에는 라벨이름으로 된 폴더가 있어야 하고, 그 안에 이미지 파일이 있으면 됩니다.
    # 예시: dataset/a/001.png, dataset/a/002.png, dataset/b/001.png, dataset/b/002.png
    # 라벨별로 최대 maxAugemtation 개수까지 이미지 augmentation을 실시합니다.
    def prepareData(self, datasetFolder, maxAugmentation=150):
        folders = []
        for d in os.listdir(datasetFolder):
            if os.path.isdir(datasetFolder + "/" + d):
                numImg = len([img for img in os.listdir(datasetFolder + "/" + d) if
                              os.path.isfile(datasetFolder + "/" + d + "/" + img)])
                if numImg < 3: continue
                folders.append((d, numImg))
        self.nClasses = len(folders)

        self.oNames = []
        data = []
        for n, (folder, numImg) in enumerate(folders):
            oname = NNOCRModel.restoreName(folder)
            self.oNames.append(oname)
            augmentation = numImg < maxAugmentation
            augRatio = maxAugmentation / numImg
            if augRatio > 21:
                degrees = list(range(-8, 9, 2))
            elif augRatio > 15:
                degrees = list(range(-8, 9, 3))
            elif augRatio > 9:
                degrees = list(range(-8, 9, 4))
            elif augRatio > 3:
                degrees = list(range(-8, 9, 8))
            else:
                degrees = [2]
            for img in os.listdir(datasetFolder + "/" + folder):
                path = datasetFolder + "/" + folder + "/" + img
                if not os.path.isfile(path): continue
                image = io.imread(path)
                image = numpy.invert(image)
                image = numpy.average(image, axis=2)
                xs = transform.resize(image, (self.nImgSize, self.nImgSize))
                xs = numpy.reshape(xs, self.nInput).astype(float)
                xs /= numpy.max(xs)
                y = numpy.zeros((self.nClasses,), dtype=float)
                y[n] = 1
                data.append((xs, y))
                if augmentation:
                    for scale in [0.9, 1.0, 1.1]:
                        for degree in degrees:
                            if scale == 1.0 and degree == 0: continue
                            if numImg >= maxAugmentation: break
                            xs = transform.resize(transform.rescale(transform.rotate(image, degree), scale),
                                                  (self.nImgSize, self.nImgSize))
                            xs = numpy.reshape(xs, self.nInput).astype(float)
                            xs /= numpy.max(xs)
                            data.append((xs, y))
                            numImg += 1
        random.seed(1)
        random.shuffle(data)

        print(self.oNames)
        print('Total images: ', len(data))
        self.xhat = numpy.asarray([d[0] for d in data])
        self.yhat = numpy.asarray([d[1] for d in data])

    # 신경망을 생성합니다.
    def buildGraph(self):
        self.x = tf.placeholder(tf.float32, [None, self.nInput])
        self.y = tf.placeholder(tf.float32, [None, self.nClasses])

        weights = {
            'h1': tf.Variable(tf.random_normal([self.nInput, self.nLayer1], stddev=0.01), dtype=tf.float32),
            'h2': tf.Variable(tf.random_normal([self.nLayer1, self.nLayer2], stddev=0.01), dtype=tf.float32),
            'out': tf.Variable(tf.random_normal([self.nLayer2, self.nClasses], stddev=0.01), dtype=tf.float32)
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.nLayer1], stddev=0.01), dtype=tf.float32),
            'b2': tf.Variable(tf.random_normal([self.nLayer2], stddev=0.01), dtype=tf.float32),
            'out': tf.Variable(tf.random_normal([self.nClasses], stddev=0.01), dtype=tf.float32)
        }
        layer_1 = tf.add(tf.matmul(self.x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.sigmoid(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.sigmoid(layer_2)

        self.pred = tf.nn.softmax(tf.matmul(layer_2, weights['out']) + biases['out'])
        self.cost = -tf.reduce_sum(self.pred * tf.log(self.y + 1e-10)) \
                    + 0.001 * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(
            weights['out']) +
                               tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(biases['b2']) + tf.nn.l2_loss(biases['out']))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=NNOCRModel.LEARNING_RATE).minimize(self.cost)

    # 전체 데이터를 train과 test 데이터로 분할하여K cross validation을 실시합니다.
    def validateKFold(self, n=5, epochs=10):#epochs
        accuracies = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(self.xhat):
            trainX, trainY = self.xhat[train], self.yhat[train]
            testX, testY = self.xhat[test], self.yhat[test]

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                for epoch in range(epochs):
                    avg_cost = 0.
                    total_batch = len(trainX) // NNOCRModel.BATCH_SIZE
                    for i in range(total_batch):
                        batchX = trainX[i * len(trainX) // total_batch: (i + 1) * len(trainX) // total_batch]
                        batchY = trainY[i * len(trainX) // total_batch: (i + 1) * len(trainX) // total_batch]
                        _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batchX, self.y: batchY})
                        avg_cost += c / total_batch
                    if epoch % NNOCRModel.DISPLAY_STEP == 0:
                        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                accu = accuracy.eval({self.x: testX, self.y: testY})
                print("Accuracy:", accu)
                accuracies.append(accu)
        return accuracies

    def _saveParams(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.nLayer1, self.nLayer2, self.nImgSize, self.nInput, self.nClasses, self.oNames), f)

    def _loadParams(self, path):
        with open(path, 'rb') as f:
            self.nLayer1, self.nLayer2, self.nImgSize, self.nInput, self.nClasses, self.oNames = pickle.load(f)

    # 전체 데이터를 이용해 train을 실시합니다.
    def train(self, epochs=10):#epochs
        trainX, trainY = self.xhat, self.yhat
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = len(trainX) // NNOCRModel.BATCH_SIZE
            for i in range(total_batch):
                batchX = trainX[i * len(trainX) // total_batch: (i + 1) * len(trainX) // total_batch]
                batchY = trainY[i * len(trainX) // total_batch: (i + 1) * len(trainX) // total_batch]
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batchX, self.y: batchY})
                avg_cost += c / total_batch
            if epoch % NNOCRModel.DISPLAY_STEP == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

    def _saveModel(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def _loadModel(self, path):
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, path)

    def save(self, path):
        self._saveParams(path + '.pickle')
        self._saveModel(path)

    @staticmethod
    def load(path):
        inst = NNOCRModel(0, 0, 0)
        inst._loadParams(path + '.pickle')
        inst.buildGraph()
        inst._loadModel(path)
        return inst

    def predictByRawImg(self, imageArr, dictType=False):
        dataArr = []
        for img in imageArr:
            img = transform.resize(img, (self.nImgSize, self.nImgSize))
            img = numpy.reshape(numpy.invert(img), self.nInput).astype(float)
            img /= max(numpy.max(img), 32)
            dataArr.append(img)
        return self.predict(numpy.array(dataArr), dictType)

    def predict(self, dataArr, dictType=False):
        predictions = self.pred.eval({self.x: dataArr}, session=self.sess)
        if dictType:
            predictions = [dict(zip(self.oNames, p)) for p in predictions]
        return predictions


if __name__ == '__main__':

    ocr = NNOCRModel(800, 200, 32)
    ocr.prepareData('C:\\Users\\이서정\\PycharmProjects\\OcrProject\\phd08_png_results1\\', 50)
    ocr.buildGraph()
    # 성능 평가는 다음과 같이
    accs = ocr.validateKFold(5, 400)
    print("Avg Accuracy: %g" % (sum(accs) / len(accs)))
    ocr.train(5)#epoch
    ocr.save('./model')

    # 로딩은 다음과 같이
    # ocr = NNOCRModel.load('./model')
    # ocr.predict(numpy.random.rand(1, 1024))