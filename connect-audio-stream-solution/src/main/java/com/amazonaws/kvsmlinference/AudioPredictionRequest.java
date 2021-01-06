package com.amazonaws.kvsmlinference;

/*
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

import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AudioPredictionRequest {

    String streamARN = null;
    String inputFileName = null;
    String startFragmentNum = null;
    String connectContactId = null;
    boolean transcriptionEnabled = false;
    Optional<Boolean> saveCallRecording = Optional.empty();
    boolean streamAudioFromCustomer = true;
    boolean streamAudioToCustomer = true;

    public String getStreamARN() {

        return this.streamARN;
    }

    public void setStreamARN(String streamARN) {

        this.streamARN = streamARN;
    }

    public String getInputFileName() {

        return this.inputFileName;
    }

    public void setInputFileName(String inputFileName) {

        this.inputFileName = inputFileName;
    }

    public String getStartFragmentNum() {

        return this.startFragmentNum;
    }

    public void setStartFragmentNum(String startFragmentNum) {

        this.startFragmentNum = startFragmentNum;
    }

    public String getConnectContactId() {

        return this.connectContactId;
    }

    public void setConnectContactId(String connectContactId) {

        this.connectContactId = connectContactId;
    }

    public void setTranscriptionEnabled(boolean enabled) {
        transcriptionEnabled = enabled;
    }

    public void setStreamAudioFromCustomer(boolean enabled) {
        streamAudioFromCustomer = enabled;
    }

    public boolean isStreamAudioFromCustomer() {
        return  streamAudioFromCustomer;
    }

    public void setStreamAudioToCustomer(boolean enabled) {
        streamAudioToCustomer = enabled;
    }

    public boolean isStreamAudioToCustomer() {
        return  streamAudioToCustomer;
    }

    public void setSaveCallRecording(boolean shouldSaveCallRecording) {

        saveCallRecording = Optional.of(shouldSaveCallRecording);
    }

    public Optional<Boolean> getSaveCallRecording() {
        return saveCallRecording;
    }

    public boolean isSaveCallRecordingEnabled() {

        return (saveCallRecording.isPresent() ? saveCallRecording.get() : false);
    }

    public String toString() {

        return String.format("streamARN=%s, startFragmentNum=%s, connectContactId=%s, saveCallRecording=%s, streamAudioFromCustomer=%s, streamAudioToCustomer=%s",
                getStreamARN(), getStartFragmentNum(), getConnectContactId(), isSaveCallRecordingEnabled(), isStreamAudioFromCustomer(), isStreamAudioToCustomer());
    }

    public void validate() throws IllegalArgumentException {

        // complain if both are provided
        if ((getStreamARN() != null) && (getInputFileName() != null))
            throw new IllegalArgumentException("At most one of streamARN or inputFileName must be provided");
        // complain if none are provided
        if ((getStreamARN() == null) && (getInputFileName() == null))
            throw new IllegalArgumentException("One of streamARN or inputFileName must be provided");
    }

}
