// Function to detect age-restricted content
function detectAgeRestrictedContent() {
    // Get the text content of the page
    var pageContent = document.body.innerText.toLowerCase();

    // List of keywords indicating age-restricted content
    var ageRestrictedKeywords = ["adult", "18+", "explicit", "mature", "restricted"];

    // Check if any of the keywords are present in the page content
    for (var i = 0; i < ageRestrictedKeywords.length; i++) {
        if (pageContent.includes(ageRestrictedKeywords[i])) {
            return true; // Age-restricted content detected
        }
    }

    return false; // No age-restricted content detected
}

// Send a message to the background script if age-restricted content is detected
if (detectAgeRestrictedContent()) {
    chrome.runtime.sendMessage({action: "ageRestrictedContentDetected"});
}
