package org.example;

import gr.uom.java.xmi.LocationInfo;
import gr.uom.java.xmi.diff.CodeRange;
import org.eclipse.jgit.lib.Repository;
import org.json.JSONArray;
import org.json.JSONObject;
import org.refactoringminer.api.*;
import org.refactoringminer.rm1.GitHistoryRefactoringMinerImpl;
import org.refactoringminer.util.GitServiceImpl;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class ExtractMethodExtractor {

    private final Repository repository;
    private JSONArray refactoringList;

    private String defaultBranch;

    public ExtractMethodExtractor(String repositoryPath, String defaultBranch) throws Exception {
        GitService gitService = new GitServiceImpl();
//        "https://github.com/danilofes/refactoring-toy-example.git"

        String tempDirectoryPath = System.getProperty("java.io.tmpdir");

        String[] parts = repositoryPath.split("/");
        String projectName = parts[parts.length - 1].split("\\.")[0];

        System.out.println("Project Name: " + projectName);

        String clonePath = Paths.get(tempDirectoryPath, projectName).toString();

        System.out.println("Clone Path: " + clonePath);

        this.repository = gitService.cloneIfNotExists(clonePath, repositoryPath);
        this.defaultBranch = defaultBranch;

    }

    private static List<CodeRange> filterRange(List<CodeRange> codeRanges) {
        List<CodeRange> filteredCodeRanges = new ArrayList<>();
        for (CodeRange codeRange : codeRanges) {
            if (codeRange.getCodeElementType() == LocationInfo.CodeElementType.METHOD_DECLARATION) {
                filteredCodeRanges.add(codeRange);
            }
        }
        return filteredCodeRanges;
    }

    public ExtractMethodExtractor identifyRefactoringInstances() throws Exception {

        GitHistoryRefactoringMiner miner = new GitHistoryRefactoringMinerImpl();
        this.refactoringList = new JSONArray();

        miner.detectAll(this.repository,this.defaultBranch, new RefactoringHandler() {
            @Override
            public void handle(String commitId, List<Refactoring> refactorings) {
                int c = 0;
                for (Refactoring ref : refactorings) {
                    if (ref.getRefactoringType() == RefactoringType.EXTRACT_OPERATION){
                        c++;
                        JSONObject refactoring = new JSONObject();
                        refactoring.put("commitID", commitId);
                        refactoring.put("leftSide", ExtractMethodExtractor.filterRange(ref.leftSide()));
                        refactoring.put("rightSide", ExtractMethodExtractor.filterRange(ref.rightSide()));
                        refactoringList.put(refactoring);
                    }
                }
                System.out.println("Commit " + commitId + ": " + c + " refactorings");
            }

        });

        return this;
    }

    public void generateSamples(String outPath){
        RepositoryParser repoParser = new RepositoryParser(this.repository);
        List<Map<String, Object>> samples = new ArrayList<>();
        try {
            for (int i = 0; i < this.refactoringList.length(); i++) {
                JSONObject refactoring = this.refactoringList.getJSONObject(i);
                Map<String, Object> instanceSample = repoParser.getSamples(refactoring, true);
                samples.add(instanceSample);
            }

            // Write to file
            JSONArray sampleArray = Utils.convertToJsonArray(samples);

            Utils.appendToFile(sampleArray, outPath);
        }
        catch (Exception e) {
            System.out.println("Exception: " + e);
        }

    }
}
