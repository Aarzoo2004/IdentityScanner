chrome.runtime.onInstalled.addListener(() => {
    console.log('Extension installed');
  });
  
  chrome.runtime.onStartup.addListener(() => {
    console.log('Extension started');
  });
  
  chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
      if (message.action === "ageRestrictedContentDetected") {
          // Handle the detection of age-restricted content
          // For example, display a warning message
          console.log('Age-restricted content detected');
      }
  });
  