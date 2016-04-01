/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package com.deeplearningserver.dependency.batik;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.deeplearningserver.util.Variables;

/**
 * Data fetcher for the MNIST dataset
 * 
 * @author Adam Gibson
 * 
 */
public class BatikDataFetcher extends BaseDataFetcher {
	public static final int NUM_EXAMPLES = 74;
	public static final int NUM_EXAMPLES_TEST = 15;

	protected transient BatikManager man;
	protected boolean binarize = true;
	protected boolean train;
	protected int[] order;
	protected Random rng;
	protected boolean shuffle;
	protected int imageWidth = 28;
	protected int imageHeight = 28;
	

	public BatikDataFetcher(boolean binarize) throws IOException {
		this(binarize, true, true, System.currentTimeMillis());
	}

	public BatikDataFetcher(boolean binarize, boolean train, boolean shuffle,
			long rngSeed) throws IOException {

//		String images = "D:\\nitip\\kuliah\\ui\\thesis\\android\\dataset batik\\final\\28x28\\";
		String labels = "";

		try {
			man = new BatikManager(Variables.PATH_DATA, labels, train, Variables.BASE_DATA_WIDTH * Variables.BASE_DATA_WIDTH);
		} catch (Exception e) {
			man = new BatikManager(Variables.PATH_DATA, labels, train, Variables.BASE_DATA_WIDTH * Variables.BASE_DATA_WIDTH);
		}

		try {
			numOutcomes = Variables.NUM_OUTCOMES;
			this.binarize = binarize;
			cursor = 0;

			inputColumns = imageWidth * imageHeight;
			this.train = train;
			this.shuffle = shuffle;

			if (train) {
				order = new int[NUM_EXAMPLES];
				totalExamples = NUM_EXAMPLES;
			} else {
				order = new int[NUM_EXAMPLES_TEST];
				totalExamples = NUM_EXAMPLES_TEST;
			}
			
			for (int i = 0; i < order.length; i++)
				order[i] = i;
			rng = new Random(rngSeed);
			reset(); // Shuffle order
		} catch (Exception e) {
			System.out.println("masukkkkkk");
			e.printStackTrace();
		}
	}

	public BatikDataFetcher() throws IOException {
		this(true);
	}

	@Override
	public void fetch(int numExamples) {
		if (!hasMore()) {
			throw new IllegalStateException(
					"Unable to getFromOrigin more; there are no more images");
		}

		// we need to ensure that we don't overshoot the number of examples
		// total
		List<DataSet> toConvert = new ArrayList<>(numExamples);
		for (int i = 0; i < numExamples; i++, cursor++) {
			if (!hasMore()) {
				break;
			}

			byte[] img = man.readImageUnsafe(order[cursor]);
			INDArray in = Nd4j.create(1, img.length);
			for (int j = 0; j < img.length; j++) {
				in.putScalar(j, ((int) img[j]) & 0xFF); // byte is loaded as
														// signed -> convert to
														// unsigned
			}

			if (binarize) {
				for (int d = 0; d < in.length(); d++) {
					if (in.getDouble(d) > 30) {
						in.putScalar(d, 1);
					} else {
						in.putScalar(d, 0);
					}
				}
			} else {
				in.divi(255);
			}

//			System.out.println("order cursor : " + man.readLabel(order[cursor]));
			INDArray out = createOutputVector(man.readLabel(order[cursor]));

			toConvert.add(new DataSet(in, out));
		}
		
		initializeCurrFromList(toConvert);
	}

	@Override
	public void reset() {
		cursor = 0;
		curr = null;
		if (shuffle)
			MathUtils.shuffleArray(order, rng);
	}

	@Override
	public DataSet next() {
		DataSet next = super.next();
		return next;
	}

}
