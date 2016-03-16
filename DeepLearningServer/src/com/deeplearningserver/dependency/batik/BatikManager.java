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


import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import com.twelvemonkeys.image.BufferedImageFactory;


public class BatikManager {
    BatikImageFile images;
    private BatikLabelFile labels;

    private byte[][] imagesArrBefore, imagesArr;
    private int[] labelsArr;
    private static final int HEADER_SIZE = 8;

    /**
     * Writes the given image in the given file using the PPM data format.
     *
     * @param image
     * @param ppmFileName
     * @throws IOException
     */
    public static void writeImageToPpm(int[][] image, String ppmFileName) throws IOException {
        try (BufferedWriter ppmOut = new BufferedWriter(new FileWriter(ppmFileName))) {
            int rows = image.length;
            int cols = image[0].length;
            ppmOut.write("P3\n");
            ppmOut.write("" + rows + " " + cols + " 255\n");
            for (int i = 0; i < rows; i++) {
                StringBuilder s = new StringBuilder();
                for (int j = 0; j < cols; j++) {
                    s.append(image[i][j] + " " + image[i][j] + " " + image[i][j] + "  ");
                }
                ppmOut.write(s.toString());
            }
        }

    }

    public BatikManager(String imagesFile, String labelsFile, boolean train, int arraySize) throws IOException {
        
    	labelsArr = new int[BatikDataFetcher.NUM_EXAMPLES];
    	imagesArr = new byte[BatikDataFetcher.NUM_EXAMPLES][arraySize];
    	imagesArrBefore = loadData(imagesFile);
    	
    	int count = 0;
    	for (byte[] dataArr : imagesArrBefore) {
    		imagesArr[count] = Arrays.copyOfRange(dataArr, 0, arraySize - 1);
    		count++;
    	}
        
        System.out.println();
    }
    
    public BatikManager(String imagesFile, String labelsFile, boolean train) throws IOException {
        
    	labelsArr = new int[BatikDataFetcher.NUM_EXAMPLES];
    	imagesArrBefore = loadData(imagesFile);
    	
    	int count = 0;
    	for (byte[] dataArr:imagesArrBefore) {
    		imagesArr[count] = Arrays.copyOfRange(dataArr, 0, 28*28);
    		count++;
    	}
        
        System.out.println();
    }

    public BatikManager(String imagesFile, String labelsFile) throws IOException{
        this(imagesFile,labelsFile,true);
    }

    /**
     * Reads the current image.
     *
     * @return matrix
     * @throws IOException
     */
    public int[][] readImage() throws IOException {
        if (images == null) {
            throw new IllegalStateException("Images file not initialized.");
        }
        return images.readImage();
    }

    public byte[] readImageUnsafe(int i){
        return imagesArr[i];
    }

    /**
     * Set the position to be read.
     *
     * @param index
     */
    public void setCurrent(int index) {
        images.setCurrentIndex(index);
        labels.setCurrentIndex(index);
    }

    /**
     * Reads the current label.
     *
     * @return int
     * @throws IOException
     */
    public int readLabel() throws IOException {
        if (labels == null) {
            throw new IllegalStateException("labels file not initialized.");
        }
        return labels.readLabel();
    }

    public int readLabel(int i){
        return labelsArr[i];
    }

    /**
     * Get the underlying images file as {@link BatikImageFile}.
     *
     * @return {@link BatikImageFile}.
     */
    public BatikImageFile getImages() {
        return images;
    }

    /**
     * Get the underlying labels file as {@link BatikLabelFile}.
     *
     * @return {@link BatikLabelFile}.
     */
    public BatikLabelFile getLabels() {
        return labels;
    }

    /**
     * Close any resources opened by the manager.
     */
    public void close() {
        if(images != null) {
            try {
                images.close();
            } catch (IOException e) {}
            images = null;
        }
        if(labels != null) {
            try {
                labels.close();
            } catch (IOException e) {}
            labels = null;
        }
    }
    
	public byte[][] loadData(String dataSetPath) {
		
		byte[][] datas = null;
		
		try {
			File dataSetFolder = new File(dataSetPath);			

			// generate array of bytes from image
			int count = 0;
			datas = new byte[dataSetFolder.listFiles().length][0];
			
			Set<String> fileNames = new HashSet<>();
			for (File f : dataSetFolder.listFiles()) {
				fileNames.add(f.getName().split("\\ ")[0]);
			}			
			Map<String, Integer> fileNamesFinal = new java.util.HashMap<>();
			for(String s: fileNames){
//				System.out.println(s + " : " + count);
				fileNamesFinal.put(s, count);
				count++;
			}			
			
			count = 0;
			for (File f : dataSetFolder.listFiles()) {
				byte[] data = convertImageToByte(f);
				datas[count] = data;
				labelsArr[count] = fileNamesFinal.get(f.getName().split("\\ ")[0]);
				count++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return datas;
	}
	
	public byte[] convertImageToByte(File f) {
		byte[] result = null;
		try {
			BufferedImage image = ImageIO.read(f);
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			ImageIO.write( image, "jpg", baos );
			baos.flush();
			result = baos.toByteArray();
			baos.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
//	public byte[] convertImageToByteNew(File f) {
//		byte[] result = null;
//		try {
//			FileInputStream fis = new FileInputStream(f);
//			ByteArrayOutputStream baos = new ByteArrayOutputStream();
//			
//			result = new byte[1024];
//			for (int readNum; (readNum = fis.read(result)) != -1;) {
//                //Writes to this byte array output stream
//				baos.write(result, 0, readNum); 
//                System.out.println("read " + readNum + " bytes,");
//            }
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		return result;
//	}
}
