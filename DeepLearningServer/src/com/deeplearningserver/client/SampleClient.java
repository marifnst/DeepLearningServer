package com.deeplearningserver.client;

import java.util.Map;

import javax.xml.namespace.QName;
import javax.xml.ws.Service;
import javax.xml.ws.soap.SOAPBinding;

import com.deeplearningserver.services.CNNServices;

public class SampleClient {
	private static final QName SERVICE_NAME = new QName("http://services.deeplearningserver.com/", "CNNServices");
	private static final QName PORT_NAME = new QName("http://services.deeplearningserver.com/", "CNNServicesPort");

	public static void main(String args[]) throws Exception {
		Service service = Service.create(SERVICE_NAME);
		// Endpoint Address
		String endpointAddress = "http://localhost:8080/DeepLearningServer/services/batik_detection";
		// If web service deployed on Tomcat (either standalone or embedded)
		// as described in sample README, endpoint should be changed to:
		// String endpointAddress =
		// "http://localhost:8080/java_first_jaxws/services/hello_world";

		// Add a port to the Service
		service.addPort(PORT_NAME, SOAPBinding.SOAP11HTTP_BINDING, endpointAddress);

		CNNServices hw = service.getPort(CNNServices.class);
		//System.out.println(hw.batikDetection());

	}
}
