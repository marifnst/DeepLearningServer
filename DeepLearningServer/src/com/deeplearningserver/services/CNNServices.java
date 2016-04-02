package com.deeplearningserver.services;

import javax.jws.WebService;

@WebService
public interface CNNServices {
	public String batikDetection(String imageFile);
}
