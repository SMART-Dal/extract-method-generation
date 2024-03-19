package org.example;

public class Main {
    public static void main(String[] args) {

        System.out.println("Hello world!");
        Refactorings refactorings = new Refactorings();
        try {
            refactorings.getRefactorings();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}