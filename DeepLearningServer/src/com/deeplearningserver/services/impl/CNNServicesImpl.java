package com.deeplearningserver.services.impl;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.UUID;

import javax.imageio.ImageIO;
import javax.jws.WebService;

import org.kobjects.base64.Base64;

import com.deeplearningserver.services.CNNServices;

@WebService(endpointInterface = "com.deeplearningserver.services.CNNServices",serviceName = "CNNServices")
public class CNNServicesImpl implements CNNServices {

	@Override
	public String batikDetection(String imageFile) {
		
		byte[] imageData = Base64.decode(imageFile);
		InputStream in = new ByteArrayInputStream(imageData);
		BufferedImage bImageFromConvert = null;
		
		try {
			bImageFromConvert = ImageIO.read(in);
			ImageIO.write(bImageFromConvert, "jpg", new File("D:\\source\\" + UUID.randomUUID().toString() + ".jpg"));
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		return "dari server";
	}

}
