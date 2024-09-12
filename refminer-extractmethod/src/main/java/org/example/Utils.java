package org.example;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Map;

public class Utils {

    public static JSONArray convertToJsonArray(List<Map<String, Object>> mapList) {
        JSONArray jsonArray = new JSONArray();
        for (Map<String, Object> map : mapList) {
            jsonArray.put(new JSONObject(map));
        }
        return jsonArray;
    }

    public static void appendToFile(JSONArray jsonArray, String filePath) {
        try (FileWriter file = new FileWriter(filePath, true)) { // true for append mode
            for (int i = 0; i < jsonArray.length(); i++) {
                file.write(jsonArray.getJSONObject(i).toString());
                file.write(System.lineSeparator()); // new line for each JSON object
            }
            System.out.println("Successfully Appended JSON Array to File...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
