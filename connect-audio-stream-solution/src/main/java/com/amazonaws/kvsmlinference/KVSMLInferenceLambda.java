package com.amazonaws.kvsmlinference;

import com.amazonaws.SdkClientException;
import com.amazonaws.auth.AWSCredentialsProvider;
import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import com.amazonaws.kinesisvideo.parser.ebml.InputStreamParserByteSource;
import com.amazonaws.kinesisvideo.parser.mkv.StreamingMkvReader;
import com.amazonaws.kinesisvideo.parser.utilities.FragmentMetadataVisitor;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.cloudwatch.AmazonCloudWatchClientBuilder;
import com.amazonaws.services.lambda.runtime.Context;
import com.amazonaws.services.lambda.runtime.RequestHandler;
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntime;
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntimeClientBuilder;
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointRequest;
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointResult;
import com.amazonaws.services.sagemakerruntime.model.AmazonSageMakerRuntimeException;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDB;
import com.amazonaws.services.dynamodbv2.AmazonDynamoDBClientBuilder;
import com.amazonaws.services.dynamodbv2.document.DynamoDB;
import com.amazonaws.services.dynamodbv2.document.Item;
import com.amazonaws.services.dynamodbv2.document.PutItemOutcome;
import org.reactivestreams.Publisher;
import org.reactivestreams.Subscriber;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import software.amazon.awssdk.auth.credentials.AwsCredentialsProvider;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.time.Instant;

/**
 * Demonstrate Amazon Connect's real-time transcription feature using AWS Kinesis Video Streams and AWS Transcribe.
 *
 * <p>Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.</p>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so.
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
public class KVSMLInferenceLambda implements RequestHandler<AudioPredictionRequest, String> {

    private static final Regions REGION = Regions.fromName(System.getenv("APP_REGION"));
    private static final String RECORDINGS_BUCKET_NAME = System.getenv("RECORDINGS_BUCKET_NAME");
    private static final String RECORDINGS_KEY_PREFIX = System.getenv("RECORDINGS_KEY_PREFIX");
    private static final boolean CONSOLE_LOG_TRANSCRIPT_FLAG = Boolean.parseBoolean(System.getenv("CONSOLE_LOG_TRANSCRIPT_FLAG"));
    private static final boolean RECORDINGS_PUBLIC_READ_ACL = Boolean.parseBoolean(System.getenv("RECORDINGS_PUBLIC_READ_ACL"));
    private static final String START_SELECTOR_TYPE = System.getenv("START_SELECTOR_TYPE");
    private static final String SM_ENDPOINT_NAME = System.getenv("SM_ENDPOINT_NAME");
    private static final String TABLE_ML_INFERENCE = System.getenv("TABLE_ML_INFERENCE");

    private static final Logger logger = LoggerFactory.getLogger(KVSMLInferenceLambda.class);
    public static final MetricsUtil metricsUtil = new MetricsUtil(AmazonCloudWatchClientBuilder.defaultClient());
    private static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");


    /**
     * Handler function for the Lambda
     *
     * @param request
     * @param context
     * @return
     */
    @Override
    public String handleRequest(AudioPredictionRequest request, Context context) {

        logger.info("received request : " + request.toString());
        logger.info("received context: " + context.toString());

        try {
            // validate the request
            request.validate();

            startKVSToPredictionStreaming(request.getStreamARN(), request.getStartFragmentNum(), request.getConnectContactId(), request.getSaveCallRecording());

            return "{ \"result\": \"Success\" }";

        } catch (Exception e) {
            logger.error("KVS to Transcribe Streaming failed with: ", e);
            return "{ \"result\": \"Failed\" }";
        }
    }

    /**
     * Starts streaming between KVS and Transcribe
     * The transcript segments are continuously saved to the Dynamo DB table
     * At end of the streaming session, the raw audio is saved as an s3 object
     *
     * @param streamARN
     * @param startFragmentNum
     * @param contactId
     * @param saveCallRecording
     * @throws Exception
     */
    private void startKVSToPredictionStreaming(String streamARN, String startFragmentNum, String contactId, Optional<Boolean> saveCallRecording) throws Exception {
        
        String streamName = streamARN.substring(streamARN.indexOf("/") + 1, streamARN.lastIndexOf("/"));

        KVSStreamTrackObject kvsStreamTrackObjectFromCustomer = getKVSStreamTrackObject(streamName, startFragmentNum, KVSUtils.TrackName.AUDIO_FROM_CUSTOMER.getName(), contactId);
        logger.info("Start to process KVS streaming and make prediction.");

        if (kvsStreamTrackObjectFromCustomer != null) {
            // get audio streaming from KVS to local file
            ByteBuffer audioBuffer = KVSUtils.getByteBufferFromStream(kvsStreamTrackObjectFromCustomer.getStreamingMkvReader(),
                kvsStreamTrackObjectFromCustomer.getFragmentVisitor(), kvsStreamTrackObjectFromCustomer.getTagProcessor(), contactId, kvsStreamTrackObjectFromCustomer.getTrackName());

            while (audioBuffer.remaining() > 0) {
                byte[] audioBytes = new byte[audioBuffer.remaining()];
                audioBuffer.get(audioBytes);
                kvsStreamTrackObjectFromCustomer.getOutputStream().write(audioBytes);
                audioBuffer = KVSUtils.getByteBufferFromStream(kvsStreamTrackObjectFromCustomer.getStreamingMkvReader(),
                        kvsStreamTrackObjectFromCustomer.getFragmentVisitor(), kvsStreamTrackObjectFromCustomer.getTagProcessor(), contactId, kvsStreamTrackObjectFromCustomer.getTrackName());
            }
            String audioFilePath = kvsStreamTrackObjectFromCustomer.getSaveAudioFilePath().toString();
            File audioFile = new File(audioFilePath);
            logger.info("file path: "+audioFilePath);
            logger.info("file size: "+audioFile.length());

            //Upload the Raw Audio file to S3
            kvsStreamTrackObjectFromCustomer.getInputStream().close();
            kvsStreamTrackObjectFromCustomer.getOutputStream().close();
            if (audioFile.length() > 0) {
                String s3path = AudioUtils.uploadRawAudio(REGION, RECORDINGS_BUCKET_NAME, RECORDINGS_KEY_PREFIX, kvsStreamTrackObjectFromCustomer.getSaveAudioFilePath().toString(), contactId, RECORDINGS_PUBLIC_READ_ACL,
                        getAWSCredentials());
                if (s3path.length()>1) {
                    logger.info("Audio file uploaded successfully to: " + s3path);
                    try {
                        //Invoke SageMaker Inference endpoint
                        AmazonSageMakerRuntime smclient = AmazonSageMakerRuntimeClientBuilder
                                .standard()
                                .withRegion(REGION)
                                .withCredentials(getAWSCredentials())
                                .build();

                        InvokeEndpointRequest invokeEndpointRequest = new InvokeEndpointRequest();
                        invokeEndpointRequest.setContentType("text/csv");
                        invokeEndpointRequest.setEndpointName(SM_ENDPOINT_NAME);
                        invokeEndpointRequest.setBody(ByteBuffer.wrap(s3path.getBytes("UTF-8")));

                        InvokeEndpointResult result = smclient.invokeEndpoint(invokeEndpointRequest);
                        String body = StandardCharsets.UTF_8.decode(result.getBody()).toString();
                        logger.info("SageMaker Inference result body: "+body);

                        //Write to DynamoDB
                        AmazonDynamoDB ddbbuilder = AmazonDynamoDBClientBuilder
                                .standard()
                                .withRegion(REGION)
                                .build();
                        DynamoDB ddbclient = new DynamoDB(ddbbuilder);
                        Instant now = Instant.now();
                        Item ddbItem = new Item()
                                .withKeyComponent("ContactId", contactId)
                                .withKeyComponent("StartTime", now.toEpochMilli())
                                .withString("predictionTime", now.toString())
                                .withString("predictionBody", body);
                        PutItemOutcome outcome = ddbclient.getTable(TABLE_ML_INFERENCE).putItem(ddbItem);
                        logger.info("DynamoDB putItem result: "+outcome.toString());

                    } catch (UnsupportedEncodingException e) {
                        logger.error("Failed to invoke SageMaker Endpoint: ", e);
                    } catch (SdkClientException e) {
                        logger.error("Failed to invoke SageMaker Endpoint: ", e);
                    } catch (Exception e) {
                        logger.error("Exception while writing to DDB: ", e);
                    }
                }
            } else {
                logger.info("Skipping upload to S3.  saveCallRecording was disabled or audio file has 0 bytes: " + kvsStreamTrackObjectFromCustomer.getSaveAudioFilePath().toString());
            }
        }
    }


    /**
     * Create all objects necessary for KVS streaming from each track
     *
     * @param streamName
     * @param startFragmentNum
     * @param trackName
     * @param contactId
     * @return
     * @throws FileNotFoundException
     */
    private KVSStreamTrackObject getKVSStreamTrackObject(String streamName, String startFragmentNum, String trackName,
                                                         String contactId) throws FileNotFoundException {
        InputStream kvsInputStream = KVSUtils.getInputStreamFromKVS(streamName, REGION, startFragmentNum, getAWSCredentials(), START_SELECTOR_TYPE);
        StreamingMkvReader streamingMkvReader = StreamingMkvReader.createDefault(new InputStreamParserByteSource(kvsInputStream));

        KVSContactTagProcessor tagProcessor = new KVSContactTagProcessor(contactId);
        FragmentMetadataVisitor fragmentVisitor = FragmentMetadataVisitor.create(Optional.of(tagProcessor));

        String fileName = String.format("%s_%s_%s.raw", contactId, DATE_FORMAT.format(new Date()), trackName);
        Path saveAudioFilePath = Paths.get("/tmp", fileName);
        FileOutputStream fileOutputStream = new FileOutputStream(saveAudioFilePath.toString());

        return new KVSStreamTrackObject(kvsInputStream, streamingMkvReader, tagProcessor, fragmentVisitor, saveAudioFilePath, fileOutputStream, trackName);
    }

    /**
     * @return AWS credentials to be used to connect to s3 (for fetching and uploading audio) and KVS
     */
    private static AWSCredentialsProvider getAWSCredentials() {
        return DefaultAWSCredentialsProviderChain.getInstance();
    }

}
