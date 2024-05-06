// Predefine directories to be processed
dirList = newArray(
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita22_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita22_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita24_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita24_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita3_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita3_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita40_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita40_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita41_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita41_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita43_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita43_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita45_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita45_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita4_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\NG004\\Akita4_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita12_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita12_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita15_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita15_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita6_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita6_OE_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita7_OD_25x_reconstructed_multi_stacked",
    "N:\\00_Exchange\\Alois\\Skeleton_analysis_full\\Decomposed_data\\11C7\\Akita7_OE_25x_reconstructed_multi_stacked"
);

setBatchMode(true);

// Iterate over each directory in the predefined list
for (d = 0; d < dirList.length; d++) {
    input = dirList[d];
    
    // Setup output directory
    output = input + "\\output\\"; // Use "\\" for Windows paths
    File.makeDirectory(output);
    if (!File.exists(output)) {
        print("Unable to create directory: " + output);
        continue; // Skip to the next directory if unable to create the output folder
    }

    list = getFileList(input);
    filenames = newArray();
    junct_array = newArray();

    // Initialize arrays for results aggregation within the current directory
    localFilenames = newArray();
    localJunctArray = newArray();

    // Iterate over each file in the directory
    for (p = 0; p < list.length; p++) {
        if (endsWith(list[p], ".tif") || endsWith(list[p], ".tiff")) { // Corrected to include ".tiff"
            file = list[p];
            open(input + "\\" + file); // Use "\\" for Windows paths

            // Image processing steps
            setOption("BlackBackground", false);
            run("Convert to Mask");
            run("Set Measurements...", "area_fraction area mean min centroid perimeter redirect=None decimal=2");
            run("Duplicate...", " ");
            run("Skeletonize");
            run("Analyze Skeleton (2D/3D)", "prune=none show");

            // Save results for individual image
            saveAs("Measurements", output + file + ".csv");

            // Calculate junctions
            junctions = 0; 
            for(i = 0; i < nResults; i++) {
                junctions += getResult("# Junctions", i);
            }
            localJunctArray = Array.concat(localJunctArray, junctions);
            localFilenames = Array.concat(localFilenames, file);

            // Clean up for the next file
            run("Close All");
            run("Clear Results");
        }
    }

    // Save aggregated results for the current directory
    for (k = 0; k < localFilenames.length; k++) {
        setResult("Filename", k, localFilenames[k]);
        setResult("Junctions", k, localJunctArray[k]);
    }
    updateResults();
    
    selectWindow("Results");
     saveAs("Results", output + "final_results.csv");

    // Clear the results table after saving to prepare for the next directory
    run("Clear Results");
    run("Close All");
    
}

setBatchMode(false);
