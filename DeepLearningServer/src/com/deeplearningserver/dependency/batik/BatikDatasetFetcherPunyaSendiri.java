package com.deeplearningserver.dependency.batik;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

public class BatikDatasetFetcherPunyaSendiri extends BaseDataFetcher {
	
	private static BatikDatasetFetcherPunyaSendiri batikDataset;
	
	public static BatikDatasetFetcherPunyaSendiri getInstance() {
		if (batikDataset == null) {
			batikDataset = new BatikDatasetFetcherPunyaSendiri();
		}
		return batikDataset;
	}
	
	public List<DataSet> loadData(String dataSetPath) {
		
		List<DataSet> datasets = new ArrayList<>();
		
		try {
			File dataSetFolder = new File(dataSetPath);
			List<String> labelName = new ArrayList<>();
			int[] labels = new int[dataSetFolder.listFiles().length];
			
			// get labels or file name
			for (File f : dataSetFolder.listFiles()) {
				labelName.add(f.getName());
			}			

			// generate array of bytes from image
			int count = 0;
			byte[][] datas = new byte[labelName.size()][0];
			
			for (File f : dataSetFolder.listFiles()) {
				byte[] data = convertImageToBye(f);
				datas[count] = convertImageToBye(f);
				
				INDArray in = Nd4j.create(1, data.length);
	            for( int j=0; j < data.length; j++ ){
	                in.putScalar(j, ((int) data[j]) & 0xFF);
	            }
	            
				RandomAccessFile raf = new RandomAccessFile(f, "r");
				labels[count] = raf.readUnsignedByte();
	            
	            INDArray out = FeatureUtil.toOutcomeVector(raf.readUnsignedByte(), 10);
	            
	            datasets.add(new DataSet(in, out));
	            
				count++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return datasets;
	}

	public byte[] convertImageToBye(File f) {
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
	
	public static void main(String args[]) {
		BatikDatasetFetcherPunyaSendiri bs = new BatikDatasetFetcherPunyaSendiri();
		bs.loadData("C:\\Users\\Emerio\\Downloads\\lfw\\");
		System.out.println("load data image success");
	}

	@Override
	public void fetch(int numExamples) {
		List<DataSet> datasets = new ArrayList<>();
		String dataSetPath = "C:\\Users\\Emerio\\Downloads\\lfw";
		
		try {
			File dataSetFolder = new File(dataSetPath);
			List<String> labelName = new ArrayList<>();
			int[] labels = new int[dataSetFolder.listFiles().length];
			
			// get labels or file name
			for (File f : dataSetFolder.listFiles()) {
				labelName.add(f.getName());
			}			

			// generate array of bytes from image
			int count = 0;
			byte[][] datas = new byte[labelName.size()][0];
			
			for (File f : dataSetFolder.listFiles()) {
				byte[] data = convertImageToBye(f);
				datas[count] = convertImageToBye(f);
				
				INDArray in = Nd4j.create(1, data.length);
	            for( int j=0; j < data.length; j++ ){
	                in.putScalar(j, ((int) data[j]) & 0xFF);
	            }
	            
				RandomAccessFile raf = new RandomAccessFile(f, "r");
				labels[count] = raf.readUnsignedByte();
	            
	            INDArray out = FeatureUtil.toOutcomeVector(raf.readUnsignedByte(), 10);
	            
	            datasets.add(new DataSet(in, out));
	            
				count++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
}
