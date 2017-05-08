__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void image_flip(
        __read_only image2d_t inImage,
        __write_only image2d_t outImage
    ) {
	
	const int2 pos = {get_global_id(0), get_global_id(1)};
	
	float4 col =  read_imagef(inImage, sampler, pos);
	write_imagef(outImage, pos, col);
	// float col = ((float) pos.x)/255.0;
	// write_imagef(outImage, pos, (float4)(col, 0.0, 0.0, 0.0));

}