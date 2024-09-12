package org.example;

import org.eclipse.jgit.lib.ObjectId;
import org.eclipse.jgit.lib.Repository;
import org.eclipse.jgit.revwalk.RevCommit;
import org.eclipse.jgit.revwalk.RevWalk;
import org.json.JSONObject;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class RepositoryParser {

    private final Repository repo;
    private String parentSHA;

    private String commitSHA;

    public RepositoryParser(Repository repo) {

        this.repo = repo;
        this.parentSHA = null;
    }


    public void getParentSHA(String commitSHA) {
        ObjectId commitId = ObjectId.fromString(commitSHA);

        try {
            RevWalk revWalk = new RevWalk(this.repo);
            RevCommit commit = revWalk.parseCommit(commitId);
            RevCommit parent = revWalk.parseCommit(commit.getParent(0).getId());
            this.parentSHA = parent.getName();
        }
        catch (Exception e) {
            System.out.println("Exception: " + e);
        }
    }


    private ContentResult getCodeBody(int index, String key, JSONObject refactoring) throws Exception {

        ExtractMethodContent emc = new ExtractMethodContent(this.repo.getDirectory().getParent());
        boolean isLeftSide = Objects.equals(key, "leftSide");
        if (isLeftSide) {
            emc.checkoutToCommit(this.parentSHA);
        }
        else {
            emc.checkoutToCommit(this.commitSHA);
        }


        String filePath = Paths.get(this.repo.getDirectory().getParent(),
                refactoring.getJSONArray(key).getJSONObject(index).getString("filePath")).toString();


        return emc.extractMethodContent(
                filePath,
                refactoring.getJSONArray(key).getJSONObject(index).getInt("startLine"),
                refactoring.getJSONArray(key).getJSONObject(index).getInt("endLine")
        );
    }

    public Map<String, Object> getSamples(JSONObject refactoring, boolean getContext) throws Exception {

        Map<String, Object> samples = new HashMap<>();

        commitSHA = refactoring.getString("commitID");
        getParentSHA(commitSHA);

        ContentResult positiveSample = this.getCodeBody(0, "leftSide", refactoring);
        int srcIndex = 0;
        int dstIndex = 1;
        if (refactoring.getJSONArray("rightSide").getJSONObject(srcIndex).getString("description").equals("source method declaration before extraction")) {
            srcIndex = 1;
            dstIndex = 0;
        }
        ContentResult extractedMethod = this.getCodeBody(srcIndex, "rightSide", refactoring);
        ContentResult srcAfterRefactoring = this.getCodeBody(dstIndex, "rightSide", refactoring);

        if (getContext) {
            samples.put("Smelly Sample", positiveSample.getClassContent()+"\n"+positiveSample.getMethodContent());
            samples.put("Extracted Method", positiveSample.getClassContent()+extractedMethod.getMethodContent());
            samples.put("Method after Refactoring", positiveSample.getClassContent()+srcAfterRefactoring.getMethodContent());
        }
        else {
            samples.put("Smelly Sample", positiveSample.getMethodContent());
            samples.put("Extracted Method", extractedMethod.getMethodContent());
            samples.put("Method after Refactoring", srcAfterRefactoring.getMethodContent());
        }

        return samples;
    }
}
